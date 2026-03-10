#!/usr/bin/env python3
"""
Training script for EquiformerV2_MatPES
-----------------------------------------
Task: Universal MLIP — energy + forces + stress
Dataset: MatPES-PBE (~400k structures)

Multi-objective loss:
  L = w_e * MAE(E) + w_f * MAE(F) + w_s * MAE(S)

Metrics reported:
  Energy MAE  [meV/atom]
  Force  MAE  [meV/Å]
  Stress MAE  [GPa]   (1 eV/Å³ = 160.2 GPa)
"""
import torch
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

import os, sys, csv, json, math, shutil
from datetime import datetime

import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.MatPES.config import config, KBAR_TO_EV_ANG3
from data_loader_matpes import get_matpes_loaders, denormalize_energy
from equiformerv2_MatPES import EquiformerV2_MatPES

# 1 eV/Å³ = 160.2 GPa
EV_ANG3_TO_GPA = 160.2176621


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def save_checkpoint(path, ckpt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def compute_matpes_loss(
    out, batch,
    criterion,
    w_energy, w_force, w_stress,
    regress_stress,
):
    """
    Compute weighted multi-objective loss.
    Returns (total_loss, energy_loss, force_loss, stress_loss).
    """
    e_pred = out['energy']
    e_tgt = batch['energy'].view_as(e_pred)   # match [B,1] shape
    e_loss = criterion(e_pred, e_tgt)

    f_pred = out['forces']
    f_tgt  = batch['forces']
    f_loss = criterion(f_pred, f_tgt)

    if regress_stress and 'stress' in out and 'stress' in batch:
        s_pred = out['stress']
        s_tgt  = batch['stress']
        s_loss = criterion(s_pred, s_tgt)
    else:
        s_loss = torch.zeros(1, device=e_loss.device)

    total = w_energy * e_loss + w_force * f_loss + w_stress * s_loss
    return total, e_loss, f_loss, s_loss


def eval_metrics(out, batch, energy_std, regress_stress):
    """
    Compute physical-unit MAEs (no grad needed).
    Returns dict with energy_mae_meV, force_mae_meV_ang, stress_mae_GPa.
    """
    with torch.no_grad():
        # Energy MAE in meV/atom
        e_mae = (out['energy'] - batch['energy']).abs().mean().item() * energy_std * 1000

        # Force MAE in meV/Å
        f_mae = (out['forces'] - batch['forces']).abs().mean().item() * 1000

        s_mae = 0.0
        if regress_stress and 'stress' in out and 'stress' in batch:
            # stress in eV/Å³ → GPa
            s_mae = (out['stress'] - batch['stress']).abs().mean().item() * EV_ANG3_TO_GPA

    return {'energy_mae_meV': e_mae, 'force_mae_meV': f_mae, 'stress_mae_GPa': s_mae}


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Config ────────────────────────────────────────────────────
    data_path           = config['data_path']
    cache_dir           = config.get('cache_dir')
    batch_size          = config.get('batch_size', 16)
    total_epochs        = config.get('epochs', 200)
    base_save_dir       = config.get('save_dir')
    use_mixed_precision = config.get('use_mixed_precision', True)
    regress_forces      = config.get('regress_forces', True)
    regress_stress      = config.get('regress_stress', True)
    regress_magmom      = config.get('regress_magmom', False)
    w_energy            = config.get('energy_loss_weight', 1.0)
    w_force             = config.get('force_loss_weight',  1.0)
    w_stress            = config.get('stress_loss_weight', 0.1)

    # ── Run dir ───────────────────────────────────────────────────
    run_start = datetime.now()
    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(base_save_dir, f"matpes_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # ── Data ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOADING MatPES DATA")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_matpes_loaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=config.get('num_workers', 4),
        train_frac=config.get('train_frac', 0.90),
        val_frac=config.get('val_frac', 0.05),
        normalize_energy=True,
        normalize_stress=False,   # stress not normalized (physical units easier)
        max_train=config.get('max_train'),
        max_val=config.get('max_val'),
        max_test=config.get('max_test'),
        cache_dir=cache_dir,
        regress_stress=regress_stress,
        regress_magmom=regress_magmom,
        random_seed=config.get('random_seed', 42),
    )

    energy_mean = train_loader.dataset.energy_mean
    energy_std  = train_loader.dataset.energy_std
    print(f"\nNorm  mean={energy_mean:.6f}  std={energy_std:.6f}  eV/atom")
    print(f"Train:{len(train_loader.dataset):,}  Val:{len(val_loader.dataset):,}  Test:{len(test_loader.dataset):,}")

    # ── Model ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    model_kwargs = {
        'use_pbc':                  config.get('use_pbc', True),
        'regress_forces':           regress_forces,
        'regress_stress':           regress_stress,
        'otf_graph':                True,
        'max_neighbors':            config.get('max_neighbors', 20),
        'max_radius':               config.get('cutoff_radius', 6.0),
        'max_num_elements':         config.get('max_num_elements', 100),
        'num_layers':               config.get('num_layers', 6),
        'sphere_channels':          config.get('sphere_channels', 128),
        'attn_hidden_channels':     config.get('attn_hidden_channels', 128),
        'num_heads':                config.get('num_heads', 8),
        'attn_alpha_channels':      config.get('attn_alpha_channels', 32),
        'attn_value_channels':      config.get('attn_value_channels', 16),
        'ffn_hidden_channels':      config.get('ffn_hidden_channels', 512),
        'norm_type':                config.get('norm_type', 'rms_norm_sh'),
        'lmax_list':                config.get('lmax_list', [4]),
        'mmax_list':                config.get('mmax_list', [2]),
        'grid_resolution':          config.get('grid_resolution', 18),
        'edge_channels':            config.get('edge_channels', 128),
        'use_atom_edge_embedding':  config.get('use_atom_edge_embedding', True),
        'share_atom_edge_embedding':config.get('share_atom_edge_embedding', False),
        'use_m_share_rad':          config.get('use_m_share_rad', False),
        'distance_function':        config.get('distance_function', 'gaussian'),
        'num_distance_basis':       config.get('num_radial_bases', 512),
        'attn_activation':          config.get('attn_activation', 'scaled_silu'),
        'use_s2_act_attn':          config.get('use_s2_act_attn', False),
        'use_attn_renorm':          config.get('use_attn_renorm', True),
        'ffn_activation':           config.get('ffn_activation', 'scaled_silu'),
        'use_gate_act':             config.get('use_gate_act', False),
        'use_grid_mlp':             config.get('use_grid_mlp', False),
        'use_sep_s2_act':           config.get('use_sep_s2_act', True),
        'alpha_drop':               config.get('alpha_drop', 0.05),
        'drop_path_rate':           config.get('drop_path_rate', 0.05),
        'proj_drop':                config.get('proj_drop', 0.0),
        'weight_init':              config.get('weight_init', 'normal'),
    }

    model = EquiformerV2_MatPES(**model_kwargs).to(device)
    print(f"Model params: {model.num_params:,}  ({model.num_params * 4 / 1e6:.1f} MB)")
    print(f"Regress forces: {regress_forces}  stress: {regress_stress}")

    # ── Resume ────────────────────────────────────────────────────
    start_epoch     = 1
    best_val_f_mae  = float('inf')
    train_losses, val_losses = [], []
    val_e_maes, val_f_maes, val_s_maes = [], [], []
    learning_rates  = []

    resume_path = config.get('resume_from')
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch    = ckpt.get('epoch', 0) + 1
        best_val_f_mae = ckpt.get('best_val_f_mae', float('inf'))
        train_losses   = ckpt.get('train_losses', [])
        val_f_maes     = ckpt.get('val_f_maes', [])
        print(f"  Resumed at epoch {start_epoch}, best force MAE: {best_val_f_mae:.3f} meV/Å")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 2e-4),
        weight_decay=config.get('weight_decay', 1e-2),
    )
    warmup_epochs = config.get('warmup_epochs', 5)
    warmup_sched  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer,
                                       T_max=max(1, total_epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_epochs])

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    scaler    = GradScaler() if use_mixed_precision else None
    criterion = nn.L1Loss()
    grad_clip = config.get('gradient_clip_norm', 5.0)

    print(f"\nLoss weights: E={w_energy}  F={w_force}  S={w_stress}")
    print(f"Mixed precision: {'ON' if use_mixed_precision else 'OFF'}")

    # ════════════════════════════════════════════════════════════
    # Training loop
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"TRAINING  epochs={total_epochs}  batch={batch_size}")
    print("=" * 60)

    for epoch in range(start_epoch, total_epochs + 1):

        # ── Train ─────────────────────────────────────────────────
        model.train()
        tloss_sum = 0.0
        n_train   = 0
        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch:03d}/{total_epochs} [Train]",
                    leave=False)

        for batch in pbar:
            # Move all tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            if use_mixed_precision:
                with autocast():
                    out = model(batch)
                    total_loss, e_loss, f_loss, s_loss = compute_matpes_loss(
                        out, batch, criterion,
                        w_energy, w_force, w_stress, regress_stress
                    )
                scaler.scale(total_loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch)
                total_loss, e_loss, f_loss, s_loss = compute_matpes_loss(
                    out, batch, criterion,
                    w_energy, w_force, w_stress, regress_stress
                )
                total_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bsz = batch['energy'].shape[0]
            tloss_sum += total_loss.item() * bsz
            n_train   += bsz
            pbar.set_postfix({
                'tot': f'{total_loss.item():.4f}',
                'E':   f'{e_loss.item():.4f}',
                'F':   f'{f_loss.item():.4f}',
                'lr':  f'{optimizer.param_groups[0]["lr"]:.2e}',
            })

        epoch_train_loss = tloss_sum / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        vloss_sum = 0.0
        ve_sum = vf_sum = vs_sum = 0.0
        n_val  = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                if use_mixed_precision:
                    with autocast():
                        out = model(batch)
                else:
                    out = model(batch)

                total_loss, e_loss, f_loss, s_loss = compute_matpes_loss(
                    out, batch, criterion,
                    w_energy, w_force, w_stress, regress_stress
                )
                bsz = batch['energy'].shape[0]
                vloss_sum += total_loss.item() * bsz
                m = eval_metrics(out, batch, energy_std, regress_stress)
                ve_sum += m['energy_mae_meV'] * bsz
                vf_sum += m['force_mae_meV']  * bsz
                vs_sum += m['stress_mae_GPa'] * bsz
                n_val  += bsz

        epoch_val_loss = vloss_sum / n_val
        epoch_ve = ve_sum / n_val
        epoch_vf = vf_sum / n_val
        epoch_vs = vs_sum / n_val
        val_losses.append(epoch_val_loss)
        val_e_maes.append(epoch_ve)
        val_f_maes.append(epoch_vf)
        val_s_maes.append(epoch_vs)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"\nEp {epoch:03d}/{total_epochs}")
        print(f"  Train loss  : {epoch_train_loss:.6f}")
        print(f"  Val   loss  : {epoch_val_loss:.6f}")
        print(f"  Val Energy  : {epoch_ve:.3f} meV/atom")
        print(f"  Val Forces  : {epoch_vf:.3f} meV/Å")
        print(f"  Val Stress  : {epoch_vs:.4f} GPa")
        print(f"  LR          : {current_lr:.2e}")

        # ── Save best (by force MAE — most discriminative) ────────
        if epoch_vf < best_val_f_mae:
            best_val_f_mae = epoch_vf
            ckpt = {
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'scaler_state_dict':   scaler.state_dict() if scaler else None,
                'train_loss':          epoch_train_loss,
                'val_loss':            epoch_val_loss,
                'best_val_f_mae':      best_val_f_mae,
                'val_e_mae':           epoch_ve,
                'val_s_mae':           epoch_vs,
                'energy_mean':         energy_mean,
                'energy_std':          energy_std,
                'train_losses':        train_losses,
                'val_losses':          val_losses,
                'val_f_maes':          val_f_maes,
                'config':              config,
            }
            save_checkpoint(os.path.join(run_dir, 'best_model_checkpoint.pt'), ckpt)
            print(f"  ✓ Best saved  Force MAE={best_val_f_mae:.3f} meV/Å")

        # ── Periodic checkpoint ───────────────────────────────────
        si = config.get('save_interval', 10)
        if si and epoch % si == 0:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'energy_mean': energy_mean,
                'energy_std':  energy_std,
                'config': config,
            }
            save_checkpoint(os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt'), ckpt)
            print(f"  ✓ Checkpoint saved")

    # ════════════════════════════════════════════════════════════
    # Test
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("TESTING BEST MODEL")
    print("=" * 60)

    best_ckpt = torch.load(
        os.path.join(run_dir, 'best_model_checkpoint.pt'), map_location=device
    )
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    te_sum = tf_sum = ts_sum = 0.0
    n_test = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_mixed_precision:
                with autocast():
                    out = model(batch)
            else:
                out = model(batch)
            bsz = batch['energy'].shape[0]
            m = eval_metrics(out, batch, energy_std, regress_stress)
            te_sum += m['energy_mae_meV'] * bsz
            tf_sum += m['force_mae_meV']  * bsz
            ts_sum += m['stress_mae_GPa'] * bsz
            n_test += bsz

    test_e = te_sum / n_test
    test_f = tf_sum / n_test
    test_s = ts_sum / n_test

    print(f"\nTest Energy MAE : {test_e:.3f} meV/atom")
    print(f"Test Force  MAE : {test_f:.3f} meV/Å")
    print(f"Test Stress MAE : {test_s:.4f} GPa")

    # ── Save artefacts ────────────────────────────────────────────
    import numpy as np
    run_end  = datetime.now()
    duration = (run_end - run_start).total_seconds()

    metrics = {
        'best_val_force_mae_meV': best_val_f_mae,
        'test_energy_mae_meV':    test_e,
        'test_force_mae_meV':     test_f,
        'test_stress_mae_GPa':    test_s,
        'energy_mean':            energy_mean,
        'energy_std':             energy_std,
        'run_duration_hours':     duration / 3600,
        'model_parameters':       model.num_params,
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, 'losses.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss',
                         'val_e_mae_meV', 'val_f_mae_meV', 'val_s_mae_GPa', 'lr'])
        for i, (tl, vl, ve, vf, vs, lr) in enumerate(
            zip(train_losses, val_losses, val_e_maes, val_f_maes, val_s_maes, learning_rates), 1
        ):
            writer.writerow([i, tl, vl, ve, vf, vs, lr])

    try:
        shutil.copy(
            os.path.join(run_dir, 'best_model_checkpoint.pt'),
            os.path.join(run_dir, 'best_model.pt')
        )
    except Exception:
        pass

    print(f"\nResults saved: {run_dir}")
    print("=" * 60)
    print("TRAINING COMPLETE ✓")
    print("=" * 60)
    print(f"Duration        : {duration / 3600:.2f} hours")
    print(f"Best Val Force  : {best_val_f_mae:.3f} meV/Å")
    print(f"Test Energy MAE : {test_e:.3f} meV/atom")
    print(f"Test Force  MAE : {test_f:.3f} meV/Å")
    print(f"Test Stress MAE : {test_s:.4f} GPa")
    print(f"\nBaseline (TensorNet-MatPES-PBE):")
    print(f"  Energy: ~3–5 meV/atom")
    print(f"  Forces: ~50–80 meV/Å")
    print(f"  Stress: ~0.3–0.6 GPa")


if __name__ == '__main__':
    main()