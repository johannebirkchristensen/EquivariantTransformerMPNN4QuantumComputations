#!/usr/bin/env python3
"""
Training script for EquiformerV2_MatPES — Energy + Forces
==========================================================

FORCE TRAINING DESIGN (why single backward pass):
--------------------------------------------------
Forces are computed as:
    forces = -grad(energy_total, pos, create_graph=True)

With create_graph=True, the forces themselves have a grad_fn that points
back through the energy computation graph to the model parameters.
This means f_loss = MAE(forces, targets) also has a grad_fn, and a single
loss.backward() correctly propagates gradients from BOTH energy and force
errors to all model weights.

The previous two-backward-pass design used create_graph=False, which made
forces plain tensors with no grad_fn — so f_loss.backward() had nothing
to differentiate and crashed with:
    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

create_graph=True is more memory-intensive (stores the second-order graph)
but is the standard correct approach for force-field training (used by
NequIP, MACE, EquiformerV2-OC20, etc.).
"""
import torch
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

import os, sys, csv, json
from datetime import datetime

import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.MatPES.config import config
from data_loader_matpes import get_matpes_loaders
from equiformerv2_MatPESv2 import EquiformerV2_MatPES


def save_checkpoint(path, ckpt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def train_step(model, batch, criterion, w_energy, w_force, grad_clip, optimizer):
    """
    Single training step — one forward pass, one backward pass.

    Key: create_graph=True so forces have a grad_fn back to model params,
    allowing f_loss to contribute gradients via the single backward call.
    """
    optimizer.zero_grad()

    # pos must be a leaf with requires_grad so autograd.grad can differentiate
    # energy w.r.t. it to get forces
    pos = batch['pos']  # already has requires_grad=True (set in main loop)

    out = model(batch)

    # create_graph=True: forces retain connection to model params via energy graph
    forces = -torch.autograd.grad(
        out['energy_total'].sum(),
        pos,
        create_graph=True,   # REQUIRED: makes f_loss differentiable w.r.t. model params
        retain_graph=True,   # keep graph for energy backward in the combined loss
    )[0]

    e_loss = criterion(out['energy'], batch['energy'].view_as(out['energy']))
    f_loss = criterion(forces, batch['forces'])

    # Single backward pass — gradients from both losses accumulate correctly
    loss = w_energy * e_loss + w_force * f_loss
    loss.backward()

    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item(), e_loss.item(), f_loss.item()


def eval_metrics(out, batch, energy_std):
    e_mae = (out['energy'] - batch['energy'].view_as(out['energy'])).abs().mean().item() * energy_std * 1000
    return e_mae


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    regress_forces = config.get('regress_forces', True)
    w_energy       = config.get('energy_loss_weight', 1.0)
    w_force        = config.get('force_loss_weight', 1.0)
    batch_size     = config.get('batch_size', 8)
    total_epochs   = config.get('epochs', 200)
    base_save_dir  = config.get('save_dir')
    grad_clip      = config.get('gradient_clip_norm', 5.0)

    run_start = datetime.now()
    run_dir   = os.path.join(base_save_dir, f"matpes_{run_start.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

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
        cache_dir=config.get('cache_dir'),
        regress_stress=False,
        regress_magmom=False,
        random_seed=config.get('random_seed', 42),
    )
    energy_mean = train_loader.dataset.energy_mean
    energy_std  = train_loader.dataset.energy_std
    print(f"Energy norm: mean={energy_mean:.4f}  std={energy_std:.4f} eV/atom")

    # ── Model ──────────────────────────────────────────────────────
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

    optimizer     = AdamW(model.parameters(), lr=config.get('learning_rate', 2e-4),
                          weight_decay=config.get('weight_decay', 1e-2))
    warmup_epochs = config.get('warmup_epochs', 5)
    scheduler     = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs)),
    ], milestones=[warmup_epochs])

    criterion   = nn.L1Loss()
    start_epoch = 1
    best_e_mae  = float('inf')
    train_losses, val_losses, val_e_maes, lrs = [], [], [], []

    resume_path = config.get('resume_from')
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_e_mae  = ckpt.get('best_val_e_mae', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    print(f"\nLoss weights: E={w_energy}  F={w_force}")
    print("Training with create_graph=True (single backward pass)")

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):

        model.train()
        tloss = 0.0
        pbar  = tqdm(train_loader, desc=f"Ep {epoch:03d}/{total_epochs} [Train]", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            # pos must be a differentiable leaf for force computation
            batch['pos'] = batch['pos'].detach().requires_grad_(True)

            total, e_l, f_l = train_step(
                model, batch, criterion, w_energy, w_force, grad_clip, optimizer
            )
            tloss += total * batch['energy'].shape[0]
            pbar.set_postfix(tot=f'{total:.4f}', E=f'{e_l:.4f}', F=f'{f_l:.4f}',
                             lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

        train_losses.append(tloss / len(train_loader.dataset))

        # ── Validation: energy only (no forces needed, saves memory) ──
        model.eval()
        vloss = ve = 0.0
        nv = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out    = model(batch)
                e_loss = criterion(out['energy'], batch['energy'].view_as(out['energy']))
                bsz    = batch['energy'].shape[0]
                vloss += e_loss.item() * bsz
                ve    += eval_metrics(out, batch, energy_std) * bsz
                nv    += bsz

        val_losses.append(vloss / nv)
        val_e_maes.append(ve / nv)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)

        print(f"\nEp {epoch:03d}/{total_epochs}  "
              f"train={train_losses[-1]:.5f}  val={val_losses[-1]:.5f}  "
              f"E_MAE={val_e_maes[-1]:.2f}meV  lr={lr:.2e}")

        if val_e_maes[-1] < best_e_mae:
            best_e_mae = val_e_maes[-1]
            save_checkpoint(os.path.join(run_dir, 'best_model_checkpoint.pt'), {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_e_mae': best_e_mae,
                'energy_mean': energy_mean, 'energy_std': energy_std,
                'train_losses': train_losses, 'val_losses': val_losses,
                'val_e_maes': val_e_maes, 'config': config,
            })
            print(f"  ✓ Best saved  E MAE={best_e_mae:.2f} meV/atom")

        si = config.get('save_interval', 10)
        if si and epoch % si == 0:
            save_checkpoint(os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt'), {
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'energy_mean': energy_mean, 'energy_std': energy_std, 'config': config,
            })
            print(f"  ✓ Checkpoint saved")

    # ── Test ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nTESTING\n" + "="*60)
    ckpt = torch.load(os.path.join(run_dir, 'best_model_checkpoint.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    te = tf = 0.0
    nt = 0
    for batch in tqdm(test_loader, desc='Testing', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['pos'] = batch['pos'].detach().requires_grad_(True)
        out    = model(batch)
        forces = -torch.autograd.grad(
            out['energy_total'].sum(), batch['pos'],
            create_graph=False, retain_graph=False,
        )[0]
        bsz = batch['energy'].shape[0]
        te += eval_metrics(out, batch, energy_std) * bsz
        tf += (forces - batch['forces']).abs().mean().item() * 1000 * bsz
        nt += bsz

    test_e = te / nt
    test_f = tf / nt
    print(f"\nTest Energy MAE : {test_e:.3f} meV/atom")
    print(f"Test Force  MAE : {test_f:.3f} meV/Å")
    print(f"\nBaseline (TensorNet/M3GNet): Energy ~3-5 meV/atom  Forces ~50-80 meV/Å")

    duration = (datetime.now() - run_start).total_seconds()
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump({'test_energy_mae_meV': test_e, 'test_force_mae_meV': test_f,
                   'best_val_e_mae_meV': best_e_mae,
                   'energy_mean': energy_mean, 'energy_std': energy_std,
                   'run_duration_hours': duration / 3600,
                   'model_parameters': model.num_params}, f, indent=2)
    with open(os.path.join(run_dir, 'losses.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'val_e_mae_meV', 'lr'])
        for i, (tl, vl, ve, lr) in enumerate(zip(train_losses, val_losses, val_e_maes, lrs), 1):
            w.writerow([i, tl, vl, ve, lr])

    print(f"\nDone in {duration/3600:.2f}h  Results: {run_dir}")


if __name__ == '__main__':
    main()