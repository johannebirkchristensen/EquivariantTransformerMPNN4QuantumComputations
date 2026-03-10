#!/usr/bin/env python3
"""
train_MatPESv2.py
==================
Training script for EquiformerV2_MatPES — Energy + Forces.

KEY FIXES OVER FIRST ATTEMPT
------------------------------
1. Batch size raised to 32 (config) → ~12,200 batches/epoch instead of 48,900.
   Each epoch now fits in ~3–5 h on an A100, so 200 epochs need ~10 jobs of
   48 h (or fewer if you get longer wall-time).

2. save_interval=1 (config) → checkpoint every epoch.
   The first run was killed mid-epoch-2 and only epoch-1's best checkpoint
   was preserved. Now every completed epoch is saved, so resubmitting with
   --resume is cheap.

3. Cache pre-built by preprocess_cache_matpes.py.
   The first run stalled because pymatgen was parsing structures on-the-fly
   during training AND trying to write a giant pickle. Now the cache must
   exist before this script is run.

4. Force MAE in test loop fixed.
   Old: (forces - targets).abs().mean() * 1000 * bsz / nt
        → double-counted per-atom averaging
   New: accumulated as sum over atoms, divided by total atoms

5. Wall-time safety checkpoint.
   The training loop catches SIGTERM (sent ~60 s before LSF kills the job)
   and saves an emergency checkpoint so the epoch's gradient updates aren't
   completely lost.

FORCE TRAINING DESIGN
---------------------
Forces are computed as:
    forces = -grad(energy_total, pos, create_graph=True)

create_graph=True keeps forces in the autograd graph so f_loss.backward()
propagates through the force computation all the way back to model weights.
A single loss.backward() on  w_e * e_loss + w_f * f_loss is sufficient.

use_mixed_precision must be False: AMP float16 does not support the
second-order autograd graph required for force training.
"""

import signal
import sys
import os
import csv
import json
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow slice in checkpoint (PyTorch ≥ 2.6 safe-globals requirement)
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.MatPES.config_cosinelearningGATA import config
from data_loader_matpes import get_matpes_loaders
from equiformerv2_MatPES_GATAV2 import EquiformerV2_MatPES


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, ckpt: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)   # atomic rename — no corrupt files on kill


def make_ckpt(epoch, model, optimizer, scheduler, best_e_mae,
              energy_mean, energy_std, train_losses, val_losses, val_e_maes):
    return {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_e_mae':       best_e_mae,
        'energy_mean':          energy_mean,
        'energy_std':           energy_std,
        'train_losses':         train_losses,
        'val_losses':           val_losses,
        'val_e_maes':           val_e_maes,
        'config':               config,
    }


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(model, batch, criterion, w_energy, w_force, grad_clip, optimizer):
    """
    Single forward + backward pass.

    pos requires_grad=True  →  edge_distance has grad  →  energy has grad
    create_graph=True       →  forces have grad_fn back to model params
    Single backward         →  both e_loss and f_loss gradients propagate correctly
    """
    optimizer.zero_grad()

    out = model(batch)

    forces_pred = -torch.autograd.grad(
        out['energy_total'].sum(),
        batch['pos'],
        create_graph=True,   # REQUIRED for f_loss to carry gradients to model
        retain_graph=True,
    )[0]

    e_loss = criterion(out['energy'], batch['energy'].view_as(out['energy']))
    f_loss = criterion(forces_pred, batch['forces'])
    loss   = w_energy * e_loss + w_force * f_loss

    loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), e_loss.item(), f_loss.item()


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, criterion, energy_std, device):
    """Energy-only validation (saves memory — forces not needed for model selection)."""
    model.eval()
    vloss = 0.0
    e_mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(batch)
            bsz   = batch['energy'].shape[0]
            e_pred = out['energy']
            e_true = batch['energy'].view_as(e_pred)
            vloss     += criterion(e_pred, e_true).item() * bsz
            # denormalise → meV/atom
            e_mae_sum += (e_pred - e_true).abs().mean().item() * energy_std * 1000 * bsz
            n         += bsz
    return vloss / n, e_mae_sum / n


# ── Test ──────────────────────────────────────────────────────────────────────

def test(model, test_loader, criterion, energy_std, device):
    """Energy + force MAE on test set using best checkpoint."""
    model.eval()
    te_sum  = 0.0   # energy MAE accumulator (meV/atom * n_structures)
    tf_sum  = 0.0   # force  MAE accumulator (meV/Å   * n_atoms)
    n_structs = 0
    n_atoms   = 0

    for batch in tqdm(test_loader, desc='Testing', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['pos'] = batch['pos'].detach().requires_grad_(True)

        out = model(batch)
        forces_pred = -torch.autograd.grad(
            out['energy_total'].sum(),
            batch['pos'],
            create_graph=False,
            retain_graph=False,
        )[0]

        bsz = batch['energy'].shape[0]
        nat = batch['forces'].shape[0]     # total atoms in this batch

        e_pred = out['energy']
        e_true = batch['energy'].view_as(e_pred)
        te_sum    += (e_pred - e_true).abs().mean().item() * energy_std * 1000 * bsz

        # Force MAE: sum of per-atom absolute errors (divided by n_atoms at the end)
        tf_sum    += (forces_pred - batch['forces']).abs().sum().item() * 1000
        n_structs += bsz
        n_atoms   += nat

    # energy MAE averaged per structure (each already per-atom inside the model)
    test_e_mae = te_sum  / n_structs       # meV/atom
    # force MAE averaged per component (3 components per atom)
    test_f_mae = tf_sum  / (n_atoms * 3)   # meV/Å
    return test_e_mae, test_f_mae


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Device ────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Config ────────────────────────────────────────────────────
    regress_forces = config.get('regress_forces', True)
    w_energy       = config.get('energy_loss_weight', 1.0)
    w_force        = config.get('force_loss_weight',  1.0)
    batch_size     = config.get('batch_size', 32)
    total_epochs   = config.get('epochs', 200)
    base_save_dir  = config.get('save_dir')
    grad_clip      = config.get('gradient_clip_norm', 5.0)
    save_interval  = config.get('save_interval', 1)
    resume_path    = config.get('resume_from', None)

    # ── Run directory ─────────────────────────────────────────────
    run_start = datetime.now()
    # If resuming, keep the same run directory so logs accumulate cleanly
    if resume_path and os.path.exists(resume_path):
        run_dir = os.path.dirname(resume_path)
        print(f"Resuming into existing run dir: {run_dir}")
    else:
        run_dir = os.path.join(base_save_dir, f"matpes_{run_start.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2, default=str)
    print(f"Run dir: {run_dir}")

    # ── Verify caches exist ───────────────────────────────────────
    cache_dir = config.get('cache_dir')
    for split in ('train', 'val', 'test'):
        cp = os.path.join(cache_dir, f'{split}_cache.pkl')
        if not os.path.exists(cp):
            print(f"\n  ERROR: Cache not found: {cp}")
            print("  Run preprocess_cache_matpes.py first (CPU job, ~60 min).")
            sys.exit(1)
    print("  All caches found ✓")

    # ── Data ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nLOADING DATA\n" + "="*60)
    train_loader, val_loader, test_loader = get_matpes_loaders(
        data_path=config['data_path'],
        batch_size=batch_size,
        num_workers=config.get('num_workers', 4),
        train_frac=config.get('train_frac', 0.90),
        val_frac=config.get('val_frac', 0.05),
        normalize_energy=True,
        normalize_stress=False,
        max_train=config.get('max_train'),
        max_val=config.get('max_val'),
        max_test=config.get('max_test'),
        cache_dir=cache_dir,
        regress_stress=False,
        regress_magmom=False,
        random_seed=config.get('random_seed', 42),
    )
    energy_mean = train_loader.dataset.energy_mean
    energy_std  = train_loader.dataset.energy_std
    print(f"Energy norm: mean={energy_mean:.4f}  std={energy_std:.4f} eV/atom")
    print(f"Batches per epoch — train={len(train_loader)}  val={len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nINITIALIZING MODEL\n" + "="*60)
    model = EquiformerV2_MatPES(
        use_pbc=config.get('use_pbc', True),
        regress_forces=regress_forces,
        regress_stress=False,
        max_neighbors=config.get('max_neighbors', 20),
        max_radius=config.get('cutoff_radius', 6.0),
        max_num_elements=config.get('max_num_elements', 100),
        num_layers=config.get('num_layers', 6),
        sphere_channels=config.get('sphere_channels', 128),
        attn_hidden_channels=config.get('attn_hidden_channels', 128),
        num_heads=config.get('num_heads', 8),
        attn_alpha_channels=config.get('attn_alpha_channels', 32),
        attn_value_channels=config.get('attn_value_channels', 16),
        ffn_hidden_channels=config.get('ffn_hidden_channels', 512),
        norm_type=config.get('norm_type', 'rms_norm_sh'),
        lmax_list=config.get('lmax_list', [4]),
        mmax_list=config.get('mmax_list', [2]),
        grid_resolution=config.get('grid_resolution', 18),
        edge_channels=config.get('edge_channels', 128),
        use_atom_edge_embedding=config.get('use_atom_edge_embedding', True),
        share_atom_edge_embedding=config.get('share_atom_edge_embedding', False),
        use_m_share_rad=config.get('use_m_share_rad', False),
        distance_function=config.get('distance_function', 'gaussian'),
        num_distance_basis=config.get('num_radial_bases', 512),
        attn_activation=config.get('attn_activation', 'scaled_silu'),
        use_s2_act_attn=config.get('use_s2_act_attn', False),
        use_attn_renorm=config.get('use_attn_renorm', True),
        ffn_activation=config.get('ffn_activation', 'scaled_silu'),
        use_gate_act=config.get('use_gate_act', False),
        use_grid_mlp=config.get('use_grid_mlp', False),
        use_sep_s2_act=config.get('use_sep_s2_act', True),
        alpha_drop=config.get('alpha_drop', 0.05),
        drop_path_rate=config.get('drop_path_rate', 0.05),
        proj_drop=config.get('proj_drop', 0.0),
        weight_init=config.get('weight_init', 'normal'),
    ).to(device)
    print(f"Params: {model.num_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=config.get('learning_rate', 2e-4),
                      weight_decay=config.get('weight_decay', 1e-2))

    # Plain cosine annealing over all epochs — no warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config.get('lr_min', 0.0),
    )

    criterion   = nn.L1Loss()
    start_epoch = 1
    best_e_mae  = float('inf')
    train_losses, val_losses, val_e_maes, lrs = [], [], [], []

    # ── Resume ────────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch  = ckpt.get('epoch', 0) + 1
        best_e_mae   = ckpt.get('best_val_e_mae', float('inf'))
        train_losses = ckpt.get('train_losses', [])
        val_losses   = ckpt.get('val_losses',   [])
        val_e_maes   = ckpt.get('val_e_maes',   [])
        # Restore energy stats from checkpoint in case loader order changed
        energy_mean  = ckpt.get('energy_mean', energy_mean)
        energy_std   = ckpt.get('energy_std',  energy_std)
        print(f"  Resumed from epoch {start_epoch - 1}  best E MAE={best_e_mae:.2f} meV/atom")
    else:
        if resume_path:
            print(f"  WARNING: resume_from={resume_path} not found — starting fresh.")

    print(f"\nLoss weights: E={w_energy}  F={w_force}")
    print("Force computation: create_graph=True (single backward pass)")
    print(f"Batch size: {batch_size}   Batches/epoch: {len(train_loader)}")
    print(f"Scheduler: CosineAnnealingLR  T_max={total_epochs}  eta_min={config.get('lr_min', 0.0)}")

    # ── SIGTERM handler: save emergency checkpoint ────────────────
    # LSF sends SIGTERM ~60 s before killing the job (TERM_RUNLIMIT).
    # We catch it, save state, then exit cleanly so the checkpoint is intact.
    _emergency_state = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'run_dir': run_dir, 'best_e_mae': best_e_mae,
        'energy_mean': energy_mean, 'energy_std': energy_std,
        'train_losses': train_losses, 'val_losses': val_losses,
        'val_e_maes': val_e_maes,
        'epoch': start_epoch - 1,   # updated each epoch
    }

    def _sigterm_handler(signum, frame):
        ep = _emergency_state['epoch']
        path = os.path.join(_emergency_state['run_dir'], f'emergency_checkpoint_epoch_{ep:03d}.pt')
        print(f"\n  ⚠ SIGTERM received — saving emergency checkpoint to {path}")
        save_checkpoint(path, make_ckpt(
            ep,
            _emergency_state['model'],
            _emergency_state['optimizer'],
            _emergency_state['scheduler'],
            _emergency_state['best_e_mae'],
            _emergency_state['energy_mean'],
            _emergency_state['energy_std'],
            _emergency_state['train_losses'],
            _emergency_state['val_losses'],
            _emergency_state['val_e_maes'],
        ))
        print(f"  Emergency checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── CSV log (append-safe for resume) ──────────────────────────
    csv_path = os.path.join(run_dir, 'losses.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_e_mae_meV', 'lr'])

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        _emergency_state['epoch'] = epoch

        # ── Train ─────────────────────────────────────────────────
        model.train()
        tloss = 0.0
        pbar  = tqdm(train_loader, desc=f"Ep {epoch:03d}/{total_epochs} [Train]", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            # pos must be a differentiable leaf for force computation
            batch['pos'] = batch['pos'].detach().requires_grad_(True)

            tot, e_l, f_l = train_step(
                model, batch, criterion, w_energy, w_force, grad_clip, optimizer
            )
            tloss += tot * batch['energy'].shape[0]
            pbar.set_postfix(tot=f'{tot:.4f}', E=f'{e_l:.4f}', F=f'{f_l:.4f}',
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        epoch_train_loss = tloss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        _emergency_state['train_losses'] = train_losses

        # ── Validate ──────────────────────────────────────────────
        epoch_val_loss, epoch_e_mae = validate(model, val_loader, criterion, energy_std, device)
        val_losses.append(epoch_val_loss)
        val_e_maes.append(epoch_e_mae)
        _emergency_state['val_losses']  = val_losses
        _emergency_state['val_e_maes']  = val_e_maes

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

        print(f"\nEp {epoch:03d}/{total_epochs}  "
              f"train={epoch_train_loss:.5f}  val={epoch_val_loss:.5f}  "
              f"E_MAE={epoch_e_mae:.2f}meV  lr={lr:.2e}")

        # ── Best model ────────────────────────────────────────────
        if epoch_e_mae < best_e_mae:
            best_e_mae = epoch_e_mae
            _emergency_state['best_e_mae'] = best_e_mae
            save_checkpoint(os.path.join(run_dir, 'best_model_checkpoint.pt'),
                            make_ckpt(epoch, model, optimizer, scheduler, best_e_mae,
                                      energy_mean, energy_std,
                                      train_losses, val_losses, val_e_maes))
            print(f"  ✓ Best saved  E MAE={best_e_mae:.2f} meV/atom")

        # ── Periodic checkpoint ───────────────────────────────────
        if save_interval and epoch % save_interval == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            save_checkpoint(ckpt_path,
                            make_ckpt(epoch, model, optimizer, scheduler, best_e_mae,
                                      energy_mean, energy_std,
                                      train_losses, val_losses, val_e_maes))
            print(f"  ✓ Checkpoint saved: {ckpt_path}")

            # Keep only the two most recent periodic checkpoints to save disk space
            # (best_model_checkpoint.pt is always kept)
            all_ckpts = sorted([
                f for f in os.listdir(run_dir)
                if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
            ])
            for old in all_ckpts[:-2]:
                try:
                    os.remove(os.path.join(run_dir, old))
                except OSError:
                    pass

        # ── CSV log ───────────────────────────────────────────────
        csv_writer.writerow([epoch, epoch_train_loss, epoch_val_loss, epoch_e_mae, lr])
        csv_file.flush()

    csv_file.close()

    # ── Test ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTESTING\n" + "="*60)
    best_ckpt_path = os.path.join(run_dir, 'best_model_checkpoint.pt')
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_e_mae, test_f_mae = test(model, test_loader, criterion, energy_std, device)

    print(f"\nTest Energy MAE : {test_e_mae:.3f} meV/atom")
    print(f"Test Force  MAE : {test_f_mae:.3f} meV/Å")
    print(f"\nBaseline (TensorNet/M3GNet on MatPES-PBE):")
    print(f"  Energy ~3–5 meV/atom   Forces ~50–80 meV/Å")

    duration = (datetime.now() - run_start).total_seconds()
    metrics  = {
        'test_energy_mae_meV': test_e_mae,
        'test_force_mae_meV':  test_f_mae,
        'best_val_e_mae_meV':  best_e_mae,
        'energy_mean':         energy_mean,
        'energy_std':          energy_std,
        'run_duration_hours':  duration / 3600,
        'model_parameters':    model.num_params,
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {duration/3600:.2f}h   Results: {run_dir}")


if __name__ == '__main__':
    main()