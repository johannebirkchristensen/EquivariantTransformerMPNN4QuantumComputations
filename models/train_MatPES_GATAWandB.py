#!/usr/bin/env python3
"""
train_MatPESv2.py
==================
Training script for EquiformerV2_MatPES — Energy + Forces.

Changes over previous version:
  1. NaN guard in train_step: skips batch without poisoning weights
  2. Linear warmup (1000 steps) → CosineAnnealingLR, stepped per-batch
     during warmup and per-epoch after
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb

try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.MatPES.config_cosinelearningMoreGATA import config
from data_loader_matpes import get_matpes_loaders
from equiformerv2_MatPES_GATAV2 import EquiformerV2_MatPES


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path: str, ckpt: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


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
    optimizer.zero_grad()

    out = model(batch)

    forces_pred = -torch.autograd.grad(
        out['energy_total'].sum(),
        batch['pos'],
        create_graph=True,
        retain_graph=True,
    )[0]

    e_loss = criterion(out['energy'], batch['energy'].view_as(out['energy']))
    f_loss = criterion(forces_pred, batch['forces'])
    loss   = w_energy * e_loss + w_force * f_loss

    # ── NaN / Inf guard ───────────────────────────────────────────
    # If loss explodes (bad batch or early-training instability), skip the
    # backward entirely so no NaN gradients are written into the weights.
    # Without this a single bad batch poisons all parameters permanently.
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return float('nan'), float('nan'), float('nan')

    loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), e_loss.item(), f_loss.item()


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, criterion, energy_std, device):
    model.eval()
    vloss = 0.0
    e_mae_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            out    = model(batch)
            bsz    = batch['energy'].shape[0]
            e_pred = out['energy']
            e_true = batch['energy'].view_as(e_pred)
            vloss     += criterion(e_pred, e_true).item() * bsz
            e_mae_sum += (e_pred - e_true).abs().mean().item() * energy_std * 1000 * bsz
            n         += bsz
    return vloss / n, e_mae_sum / n


# ── Test ──────────────────────────────────────────────────────────────────────

def test(model, test_loader, criterion, energy_std, device):
    model.eval()
    te_sum    = 0.0
    tf_sum    = 0.0
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
        nat = batch['forces'].shape[0]

        e_pred = out['energy']
        e_true = batch['energy'].view_as(e_pred)
        te_sum    += (e_pred - e_true).abs().mean().item() * energy_std * 1000 * bsz
        tf_sum    += (forces_pred - batch['forces']).abs().sum().item() * 1000
        n_structs += bsz
        n_atoms   += nat

    test_e_mae = te_sum / n_structs
    test_f_mae = tf_sum / (n_atoms * 3)
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
    grad_clip      = config.get('gradient_clip_norm', 1.0)
    save_interval  = config.get('save_interval', 1)
    resume_path    = config.get('resume_from', None)
    warmup_steps   = config.get('warmup_steps', 1000)

    # ── W&B init ──────────────────────────────────────────────────
    wandb_run_id = config.get('wandb_run_id', None)
    run = wandb.init(
        project='Master_Project_model_comparisons',
        entity='jbirkc-danmarks-tekniske-universitet-dtu',
        config=config,
        name=config.get('wandb_run_name', 'equiformerv2-gata-matpes_lmax6'),
        tags=['equiformerv2', 'gata', 'htr', 'matpes-pbe'],
        id=wandb_run_id,
        resume='allow',
    )

    # ── Run directory ─────────────────────────────────────────────
    run_start = datetime.now()
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

    wandb.log({
        'data/energy_mean':     energy_mean,
        'data/energy_std':      energy_std,
        'data/n_train_batches': len(train_loader),
        'data/n_val_batches':   len(val_loader),
    })

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
    wandb.log({'model/n_params': model.num_params})

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 5e-5),
        weight_decay=config.get('weight_decay', 1e-2),
    )

    # ── Scheduler: linear warmup → cosine ─────────────────────────
    # Phase 1 (steps 0 → warmup_steps): lr ramps from ~0 to target lr.
    #   Prevents large random-init gradients from exploding before the
    #   model has seen any data.
    # Phase 2 (epochs warmup_steps → end): standard cosine decay.
    #   scheduler.step() must be called every BATCH in phase 1 and
    #   every EPOCH in phase 2 — see training loop below.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config.get('lr_min', 0.0),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )
    print(f"Scheduler: {warmup_steps}-step linear warmup → "
          f"CosineAnnealingLR T_max={total_epochs} eta_min={config.get('lr_min', 0.0)}")
    print(f"Loss weights: E={w_energy}  F={w_force}  |  grad_clip={grad_clip}")

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
        energy_mean  = ckpt.get('energy_mean', energy_mean)
        energy_std   = ckpt.get('energy_std',  energy_std)
        print(f"  Resumed from epoch {start_epoch - 1}  best E MAE={best_e_mae:.2f} meV/atom")
    else:
        if resume_path:
            print(f"  WARNING: resume_from={resume_path} not found — starting fresh.")

    # ── SIGTERM handler ───────────────────────────────────────────
    _emergency_state = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'run_dir': run_dir, 'best_e_mae': best_e_mae,
        'energy_mean': energy_mean, 'energy_std': energy_std,
        'train_losses': train_losses, 'val_losses': val_losses,
        'val_e_maes': val_e_maes,
        'epoch': start_epoch - 1,
    }

    def _sigterm_handler(signum, frame):
        ep   = _emergency_state['epoch']
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
        wandb.log({'emergency_checkpoint_epoch': ep})
        wandb.finish()
        print("  Emergency checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── CSV log ───────────────────────────────────────────────────
    csv_path   = os.path.join(run_dir, 'losses.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss',
                             'val_e_mae_meV', 'lr', 'nan_batches'])

    # ── global step counter ───────────────────────────────────────
    global_step = (start_epoch - 1) * len(train_loader)

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        _emergency_state['epoch'] = epoch

        model.train()
        tloss       = 0.0
        nan_batches = 0
        n_valid     = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{total_epochs} [Train]", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['pos'] = batch['pos'].detach().requires_grad_(True)

            tot, e_l, f_l = train_step(
                model, batch, criterion, w_energy, w_force, grad_clip, optimizer
            )
            global_step += 1

            # warmup scheduler steps every batch; cosine steps every epoch
            if global_step <= warmup_steps:
                scheduler.step()

            is_nan = (tot != tot)   # NaN != NaN is True
            if is_nan:
                nan_batches += 1
                pbar.set_postfix(tot='NaN', skipped=nan_batches,
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            else:
                tloss   += tot * batch['energy'].shape[0]
                n_valid += batch['energy'].shape[0]
                pbar.set_postfix(tot=f'{tot:.4f}', E=f'{e_l:.4f}', F=f'{f_l:.4f}',
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            if global_step % 50 == 0:
                log_dict = {'train/lr': optimizer.param_groups[0]['lr'],
                            'train/nan_batch': float(is_nan)}
                if not is_nan:
                    log_dict.update({
                        'train/loss_step':   tot,
                        'train/e_loss_step': e_l,
                        'train/f_loss_step': f_l,
                    })
                wandb.log(log_dict, step=global_step)

        epoch_train_loss = tloss / max(n_valid, 1)
        train_losses.append(epoch_train_loss)
        _emergency_state['train_losses'] = train_losses

        if nan_batches > 0:
            print(f"\n  ⚠ {nan_batches} NaN batches skipped this epoch")
            wandb.log({'train/nan_batches_epoch': nan_batches}, step=global_step)

        # ── Validate ──────────────────────────────────────────────
        epoch_val_loss, epoch_e_mae = validate(model, val_loader, criterion, energy_std, device)
        val_losses.append(epoch_val_loss)
        val_e_maes.append(epoch_e_mae)
        _emergency_state['val_losses'] = val_losses
        _emergency_state['val_e_maes'] = val_e_maes

        # cosine scheduler steps every epoch (after warmup is complete)
        if global_step > warmup_steps:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

        print(f"\nEp {epoch:03d}/{total_epochs}  "
              f"train={epoch_train_loss:.5f}  val={epoch_val_loss:.5f}  "
              f"E_MAE={epoch_e_mae:.2f}meV  lr={lr:.2e}"
              + (f"  [NaN: {nan_batches}]" if nan_batches else ""))

        wandb.log({
            'train/loss_epoch': epoch_train_loss,
            'val/loss':         epoch_val_loss,
            'val/e_mae_meV':    epoch_e_mae,
            'lr':               lr,
            'epoch':            epoch,
        }, step=global_step)

        # ── Best model ────────────────────────────────────────────
        if epoch_e_mae < best_e_mae:
            best_e_mae = epoch_e_mae
            _emergency_state['best_e_mae'] = best_e_mae
            best_path = os.path.join(run_dir, 'best_model_checkpoint.pt')
            save_checkpoint(best_path, make_ckpt(
                epoch, model, optimizer, scheduler, best_e_mae,
                energy_mean, energy_std, train_losses, val_losses, val_e_maes
            ))
            print(f"  ✓ Best saved  E MAE={best_e_mae:.2f} meV/atom")
            wandb.log({'val/best_e_mae_meV': best_e_mae}, step=global_step)
            artifact = wandb.Artifact(
                name='best-model', type='model',
                description=f'Best val E MAE = {best_e_mae:.3f} meV/atom at epoch {epoch}',
            )
            artifact.add_file(best_path)
            run.log_artifact(artifact)

        # ── Periodic checkpoint ───────────────────────────────────
        if save_interval and epoch % save_interval == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            save_checkpoint(ckpt_path, make_ckpt(
                epoch, model, optimizer, scheduler, best_e_mae,
                energy_mean, energy_std, train_losses, val_losses, val_e_maes
            ))
            print(f"  ✓ Checkpoint saved: {ckpt_path}")
            all_ckpts = sorted([
                f for f in os.listdir(run_dir)
                if f.startswith('checkpoint_epoch_') and f.endswith('.pt')
            ])
            for old in all_ckpts[:-2]:
                try:
                    os.remove(os.path.join(run_dir, old))
                except OSError:
                    pass

        csv_writer.writerow([epoch, epoch_train_loss, epoch_val_loss,
                             epoch_e_mae, lr, nan_batches])
        csv_file.flush()

    csv_file.close()

    # ── Test ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTESTING\n" + "="*60)
    ckpt = torch.load(os.path.join(run_dir, 'best_model_checkpoint.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_e_mae, test_f_mae = test(model, test_loader, criterion, energy_std, device)

    print(f"\nTest Energy MAE : {test_e_mae:.3f} meV/atom")
    print(f"Test Force  MAE : {test_f_mae:.3f} meV/Å")
    print(f"\nBaseline (TensorNet/M3GNet on MatPES-PBE):")
    print(f"  Energy ~3–5 meV/atom   Forces ~50–80 meV/Å")

    duration = (datetime.now() - run_start).total_seconds()
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'test_energy_mae_meV': test_e_mae,
            'test_force_mae_meV':  test_f_mae,
            'best_val_e_mae_meV':  best_e_mae,
            'energy_mean':         energy_mean,
            'energy_std':          energy_std,
            'run_duration_hours':  duration / 3600,
            'model_parameters':    model.num_params,
        }, f, indent=2)

    wandb.log({
        'test/e_mae_meV':     test_e_mae,
        'test/f_mae_meV':     test_f_mae,
        'run_duration_hours': duration / 3600,
    })
    wandb.finish()
    print(f"\nDone in {duration/3600:.2f}h   Results: {run_dir}")


if __name__ == '__main__':
    main()