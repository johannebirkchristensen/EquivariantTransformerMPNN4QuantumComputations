#!/usr/bin/env python3
"""
resume_MatPES.py
================
Resume training EquiformerV2_MatPES from a saved checkpoint.

Usage
-----
Edit CHECKPOINT_PATH below (or pass --checkpoint on the command line),
then submit via train_matpes_resume.sh.

FIX (2026-03): Never call SequentialLR.step() after resume — it desyncs the
internal last_epoch counter and causes LR spikes at the milestone boundary.
Instead, warmup_sched and cosine_sched are used directly at all times.
Checkpoints now save 'warmup_sched_state' and 'cosine_sched_state' separately.
"""

import argparse
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.MatPES.config import config
from data_loader_matpes import get_matpes_loaders
from equiformerv2_MatPESv2 import EquiformerV2_MatPES


# ── Edit this or pass --checkpoint <path> on the command line ────────────────
DEFAULT_CHECKPOINT = (
    '/work3/s203788/Master_Project_2026/'
    'EquivariantTransformerMPNN4QuantumComputations/models/'
    'trained_models/MatPES/matpes_20260226_075919/'
    'best_model_checkpoint.pt'
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, ckpt: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def make_ckpt(epoch, model, optimizer, warmup_sched, cosine_sched,
              best_e_mae, energy_mean, energy_std, train_losses, val_losses, val_e_maes):
    return {
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Save sub-schedulers individually — never SequentialLR
        'warmup_sched_state':   warmup_sched.state_dict(),
        'cosine_sched_state':   cosine_sched.state_dict(),
        'best_val_e_mae':       best_e_mae,
        'energy_mean':          energy_mean,
        'energy_std':           energy_std,
        'train_losses':         train_losses,
        'val_losses':           val_losses,
        'val_e_maes':           val_e_maes,
        'config':               config,
    }


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
    loss.backward()
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.item(), e_loss.item(), f_loss.item()


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to checkpoint .pt file to resume from')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load checkpoint ───────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    start_epoch   = ckpt['epoch'] + 1
    best_e_mae    = ckpt.get('best_val_e_mae', float('inf'))
    energy_mean   = ckpt['energy_mean']
    energy_std    = ckpt['energy_std']
    train_losses  = ckpt.get('train_losses', [])
    val_losses    = ckpt.get('val_losses',   [])
    val_e_maes    = ckpt.get('val_e_maes',   [])
    saved_config  = ckpt.get('config', config)

    print(f"  Resuming from epoch {start_epoch}  "
          f"(best E MAE so far = {best_e_mae:.2f} meV/atom)")
    print(f"  Energy norm: mean={energy_mean:.4f}  std={energy_std:.4f} eV/atom")

    total_epochs = saved_config.get('epochs', 200)
    if start_epoch > total_epochs:
        print(f"  Already completed all {total_epochs} epochs. Nothing to do.")
        sys.exit(0)

    # ── Config values ─────────────────────────────────────────────
    w_energy      = saved_config.get('energy_loss_weight', 1.0)
    w_force       = saved_config.get('force_loss_weight',  1.0)
    batch_size    = saved_config.get('batch_size', 8)
    grad_clip     = saved_config.get('gradient_clip_norm', 5.0)
    save_interval = saved_config.get('save_interval', 1)
    warmup_epochs = saved_config.get('warmup_epochs', 5)
    cache_dir     = saved_config.get('cache_dir')

    # ── Run directory: reuse the same folder as the checkpoint ────
    run_dir = os.path.dirname(checkpoint_path)
    print(f"  Run dir: {run_dir}")

    # ── Verify caches ─────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        cp = os.path.join(cache_dir, f'{split}_cache.pkl')
        if not os.path.exists(cp):
            print(f"ERROR: Cache not found: {cp}")
            print("Run preprocess_cache_matpes.py first.")
            sys.exit(1)
    print("  Caches found ✓")

    # ── Data ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nLOADING DATA\n" + "="*60)
    train_loader, val_loader, test_loader = get_matpes_loaders(
        data_path=saved_config['data_path'],
        batch_size=batch_size,
        num_workers=saved_config.get('num_workers', 4),
        train_frac=saved_config.get('train_frac', 0.90),
        val_frac=saved_config.get('val_frac', 0.05),
        normalize_energy=True,
        normalize_stress=False,
        max_train=saved_config.get('max_train'),
        max_val=saved_config.get('max_val'),
        max_test=saved_config.get('max_test'),
        cache_dir=cache_dir,
        regress_stress=False,
        regress_magmom=False,
        random_seed=saved_config.get('random_seed', 42),
    )
    train_loader.dataset.energy_mean = energy_mean
    train_loader.dataset.energy_std  = energy_std
    print(f"Batches per epoch — train={len(train_loader)}  val={len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nINITIALIZING MODEL\n" + "="*60)
    model = EquiformerV2_MatPES(
        use_pbc=saved_config.get('use_pbc', True),
        regress_forces=saved_config.get('regress_forces', True),
        regress_stress=False,
        max_neighbors=saved_config.get('max_neighbors', 20),
        max_radius=saved_config.get('cutoff_radius', 6.0),
        max_num_elements=saved_config.get('max_num_elements', 100),
        num_layers=saved_config.get('num_layers', 6),
        sphere_channels=saved_config.get('sphere_channels', 128),
        attn_hidden_channels=saved_config.get('attn_hidden_channels', 128),
        num_heads=saved_config.get('num_heads', 8),
        attn_alpha_channels=saved_config.get('attn_alpha_channels', 32),
        attn_value_channels=saved_config.get('attn_value_channels', 16),
        ffn_hidden_channels=saved_config.get('ffn_hidden_channels', 512),
        norm_type=saved_config.get('norm_type', 'rms_norm_sh'),
        lmax_list=saved_config.get('lmax_list', [4]),
        mmax_list=saved_config.get('mmax_list', [2]),
        grid_resolution=saved_config.get('grid_resolution', 18),
        edge_channels=saved_config.get('edge_channels', 128),
        use_atom_edge_embedding=saved_config.get('use_atom_edge_embedding', True),
        share_atom_edge_embedding=saved_config.get('share_atom_edge_embedding', False),
        use_m_share_rad=saved_config.get('use_m_share_rad', False),
        distance_function=saved_config.get('distance_function', 'gaussian'),
        num_distance_basis=saved_config.get('num_radial_bases', 512),
        attn_activation=saved_config.get('attn_activation', 'scaled_silu'),
        use_s2_act_attn=saved_config.get('use_s2_act_attn', False),
        use_attn_renorm=saved_config.get('use_attn_renorm', True),
        ffn_activation=saved_config.get('ffn_activation', 'scaled_silu'),
        use_gate_act=saved_config.get('use_gate_act', False),
        use_grid_mlp=saved_config.get('use_grid_mlp', False),
        use_sep_s2_act=saved_config.get('use_sep_s2_act', True),
        alpha_drop=saved_config.get('alpha_drop', 0.05),
        drop_path_rate=saved_config.get('drop_path_rate', 0.05),
        proj_drop=saved_config.get('proj_drop', 0.0),
        weight_init=saved_config.get('weight_init', 'normal'),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Params: {model.num_params:,}  — weights loaded ✓")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=saved_config.get('learning_rate', 2e-4),
        weight_decay=saved_config.get('weight_decay', 1e-2),
    )
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # ── Schedulers: always use sub-schedulers directly ────────────
    # NEVER use SequentialLR when resuming — its internal last_epoch
    # counter desyncs and causes LR spikes at the milestone boundary.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=saved_config.get('lr_min', 0.0),
    )

    # Try loading saved sub-scheduler states (new format)
    warmup_state = ckpt.get('warmup_sched_state')
    cosine_state = ckpt.get('cosine_sched_state')

    if warmup_state is not None and cosine_state is not None:
        warmup_sched.load_state_dict(warmup_state)
        cosine_sched.load_state_dict(cosine_state)
        print("Optimizer & sub-scheduler states loaded ✓")
    else:
        # Old checkpoint had SequentialLR — fast-forward manually instead.
        print("  WARNING: No sub-scheduler states found. Fast-forwarding from epoch count.")
        epochs_in_warmup = min(start_epoch - 1, warmup_epochs)
        for _ in range(epochs_in_warmup):
            warmup_sched.step()
        if start_epoch - 1 > warmup_epochs:
            for _ in range(start_epoch - 1 - warmup_epochs):
                cosine_sched.step()
        print(f"  Scheduler fast-forwarded — LR = {optimizer.param_groups[0]['lr']:.4e}")
        print("Optimizer state loaded ✓  (scheduler reconstructed from epoch count)")

    print(f"  Starting LR: {optimizer.param_groups[0]['lr']:.4e}")

    criterion = nn.L1Loss()

    print(f"\nResuming epoch {start_epoch} → {total_epochs}")
    print(f"Loss weights: E={w_energy}  F={w_force}")
    print(f"Batch size: {batch_size}   Batches/epoch: {len(train_loader)}")

    # ── SIGTERM handler ───────────────────────────────────────────
    _state = {
        'model': model, 'optimizer': optimizer,
        'warmup_sched': warmup_sched, 'cosine_sched': cosine_sched,
        'run_dir': run_dir, 'best_e_mae': best_e_mae,
        'energy_mean': energy_mean, 'energy_std': energy_std,
        'train_losses': train_losses, 'val_losses': val_losses,
        'val_e_maes': val_e_maes, 'epoch': start_epoch - 1,
    }

    def _sigterm_handler(signum, frame):
        ep = _state['epoch']
        path = os.path.join(_state['run_dir'], f'emergency_checkpoint_epoch_{ep:03d}.pt')
        print(f"\n⚠ SIGTERM — saving emergency checkpoint: {path}")
        save_checkpoint(path, make_ckpt(
            ep, _state['model'], _state['optimizer'],
            _state['warmup_sched'], _state['cosine_sched'],
            _state['best_e_mae'], _state['energy_mean'], _state['energy_std'],
            _state['train_losses'], _state['val_losses'], _state['val_e_maes'],
        ))
        print("Emergency checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── CSV log (append) ──────────────────────────────────────────
    csv_path   = os.path.join(run_dir, 'losses.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_e_mae_meV', 'lr'])

    run_start = datetime.now()

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        _state['epoch'] = epoch

        model.train()
        tloss = 0.0
        pbar  = tqdm(train_loader, desc=f"Ep {epoch:03d}/{total_epochs} [Train]", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['pos'] = batch['pos'].detach().requires_grad_(True)
            tot, e_l, f_l = train_step(
                model, batch, criterion, w_energy, w_force, grad_clip, optimizer
            )
            tloss += tot * batch['energy'].shape[0]
            pbar.set_postfix(tot=f'{tot:.4f}', E=f'{e_l:.4f}', F=f'{f_l:.4f}',
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        epoch_train_loss = tloss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        _state['train_losses'] = train_losses

        # Validate
        epoch_val_loss, epoch_e_mae = validate(
            model, val_loader, criterion, energy_std, device
        )
        val_losses.append(epoch_val_loss)
        val_e_maes.append(epoch_e_mae)
        _state['val_losses'] = val_losses
        _state['val_e_maes'] = val_e_maes

        # ── Step the correct sub-scheduler ────────────────────────
        # Call sub-schedulers directly — never SequentialLR.step()
        if epoch <= warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        lr = optimizer.param_groups[0]['lr']

        print(f"\nEp {epoch:03d}/{total_epochs}  "
              f"train={epoch_train_loss:.5f}  val={epoch_val_loss:.5f}  "
              f"E_MAE={epoch_e_mae:.2f}meV  lr={lr:.2e}")

        # Best model
        if epoch_e_mae < best_e_mae:
            best_e_mae = epoch_e_mae
            _state['best_e_mae'] = best_e_mae
            save_checkpoint(
                os.path.join(run_dir, 'best_model_checkpoint.pt'),
                make_ckpt(epoch, model, optimizer, warmup_sched, cosine_sched,
                          best_e_mae, energy_mean, energy_std,
                          train_losses, val_losses, val_e_maes)
            )
            print(f"  ✓ Best saved  E MAE={best_e_mae:.2f} meV/atom")

        # Periodic checkpoint
        if save_interval and epoch % save_interval == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            save_checkpoint(
                ckpt_path,
                make_ckpt(epoch, model, optimizer, warmup_sched, cosine_sched,
                          best_e_mae, energy_mean, energy_std,
                          train_losses, val_losses, val_e_maes)
            )
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

        csv_writer.writerow([epoch, epoch_train_loss, epoch_val_loss, epoch_e_mae, lr])
        csv_file.flush()

    csv_file.close()

    duration = (datetime.now() - run_start).total_seconds()
    print(f"\nTraining complete in {duration/3600:.2f}h")
    print(f"Best val E MAE: {best_e_mae:.2f} meV/atom")
    print(f"Results in: {run_dir}")
    print("\nNow run test_MatPES.py to evaluate on the test set.")


if __name__ == '__main__':
    main()