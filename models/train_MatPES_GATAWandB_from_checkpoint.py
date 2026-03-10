#!/usr/bin/env python3
"""
resume_MatPES_GATA.py
=====================
Resume training EquiformerV2_MatPES with GATA + HTR from a saved checkpoint.

Usage
-----
    python resume_MatPES_GATA.py --checkpoint <path/to/checkpoint.pt>

Or edit DEFAULT_CHECKPOINT below and submit via your LSF job script.

Differences from resume_MatPES.py (original)
---------------------------------------------
- Imports equiformerv2_MatPES_GATAV2 and config_cosinelearningMoreGATA
- Scheduler: linear warmup (warmup_steps batches) → CosineAnnealingLR
  stepped per BATCH during warmup, per EPOCH after.
  global_step is restored from checkpoint so warmup is not repeated on resume.
- NaN guard in train_step: skips bad batches without poisoning weights.
  nan_batches tracked per epoch, logged to CSV and W&B.
- W&B: re-attaches to the original run via wandb_run_id stored in checkpoint.
- Loss weights / grad clip: matches GATA defaults (w_force=0.1, clip=1.0).

EXIT CODE 1 DEBUGGING
---------------------
    cat trained_models/MatPES/<run_dir>/<jobid>.err
Common causes:
  - NaN loss — check nan_batches column in losses.csv; if persistent,
    lower learning_rate or force_loss_weight further in config.
  - OOM — reduce batch_size.
  - DataLoader worker crash — check for import errors in .err.
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


# ── Edit this or pass --checkpoint on the command line ───────────────────────
DEFAULT_CHECKPOINT = (
    '/work3/s203788/Master_Project_2026/'
    'EquivariantTransformerMPNN4QuantumComputations/models/'
    'trained_models/MatPES/matpes_<timestamp>/'
    'best_model_checkpoint.pt'
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, ckpt: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def make_ckpt(epoch, global_step, model, optimizer, scheduler, best_e_mae,
              energy_mean, energy_std, train_losses, val_losses, val_e_maes,
              wandb_run_id=None):
    return {
        'epoch':                epoch,
        'global_step':          global_step,      # ← needed to resume warmup correctly
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
        'wandb_run_id':         wandb_run_id,
    }


def train_step(model, batch, criterion, w_energy, w_force, grad_clip, optimizer):
    """Forward + backward with NaN guard."""
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

    # NaN guard — skip batch without corrupting weights
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return float('nan'), float('nan'), float('nan')

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
    return te_sum / n_structs, tf_sum / (n_atoms * 3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to .pt checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override total epochs (default: use value saved in checkpoint)')
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
    global_step   = ckpt.get('global_step', 0)   # 0 if old checkpoint pre-dates this field
    best_e_mae    = ckpt.get('best_val_e_mae', float('inf'))
    energy_mean   = ckpt['energy_mean']
    energy_std    = ckpt['energy_std']
    train_losses  = ckpt.get('train_losses', [])
    val_losses    = ckpt.get('val_losses',   [])
    val_e_maes    = ckpt.get('val_e_maes',   [])
    wandb_run_id  = ckpt.get('wandb_run_id', None)
    saved_config  = ckpt.get('config', config)

    print(f"  Resumed from epoch {start_epoch - 1}  "
          f"(best E MAE = {best_e_mae:.2f} meV/atom  global_step = {global_step})")
    print(f"  Energy norm: mean={energy_mean:.4f}  std={energy_std:.4f} eV/atom")

    total_epochs = args.epochs if args.epochs else saved_config.get('epochs', 200)
    if start_epoch > total_epochs:
        print(f"  Already completed all {total_epochs} epochs. Nothing to do.")
        sys.exit(0)

    # ── Config values ─────────────────────────────────────────────
    w_energy      = saved_config.get('energy_loss_weight', 1.0)
    w_force       = saved_config.get('force_loss_weight',  0.1)
    batch_size    = saved_config.get('batch_size', 32)
    grad_clip     = saved_config.get('gradient_clip_norm', 1.0)
    save_interval = saved_config.get('save_interval', 1)
    warmup_steps  = saved_config.get('warmup_steps', 1000)
    cache_dir     = saved_config.get('cache_dir')

    # ── Run directory: reuse same folder as checkpoint ────────────
    run_dir = os.path.dirname(checkpoint_path)
    print(f"  Run dir: {run_dir}")

    # ── W&B: re-attach to original run ───────────────────────────
    run = wandb.init(
        project='Master_Project',
        entity='jbirkc-danmarks-tekniske-universitet-dtu',
        config=dict(saved_config),
        name=saved_config.get('wandb_run_name', 'equiformerv2-gata-htr'),
        tags=['equiformerv2', 'gata', 'htr', 'matpes-pbe', 'resume'],
        id=wandb_run_id,      # None → new run; existing id → continues old run
        resume='allow',
    )
    wandb_run_id = run.id    # update in case it was None

    # ── Verify caches ─────────────────────────────────────────────
    for split in ('train', 'val', 'test'):
        cp = os.path.join(cache_dir, f'{split}_cache.pkl')
        if not os.path.exists(cp):
            print(f"ERROR: Cache not found: {cp}")
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
    # Use energy stats from checkpoint — NOT recomputed — to ensure consistency
    train_loader.dataset.energy_mean = energy_mean
    train_loader.dataset.energy_std  = energy_std
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
    wandb.log({'model/n_params': model.num_params})

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=saved_config.get('learning_rate', 5e-5),
        weight_decay=saved_config.get('weight_decay', 1e-2),
    )

    # ── Scheduler: warmup → cosine ────────────────────────────────
    # Reconstruct the same SequentialLR used during original training.
    # global_step tells us whether we are still inside the warmup window
    # or have already passed it — the training loop uses this to decide
    # whether to call scheduler.step() per-batch or per-epoch.
    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=saved_config.get('lr_min', 0.0),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Scheduler state: load if compatible, otherwise fast-forward.
    # Incompatibility happens if the checkpoint was saved with a plain
    # CosineAnnealingLR (no warmup) and we are now using SequentialLR.
    sched_state  = ckpt.get('scheduler_state_dict')
    loaded_sched = False
    if sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
            loaded_sched = True
            print("Optimizer & scheduler state loaded ✓")
        except (KeyError, ValueError) as e:
            print(f"  WARNING: could not load scheduler state ({e}).")
            print(f"  Fast-forwarding scheduler by {global_step} steps instead.")

    if not loaded_sched:
        # Fast-forward through warmup steps already completed
        steps_in_warmup = min(global_step, warmup_steps)
        for _ in range(steps_in_warmup):
            warmup_sched.step()
        # Then advance cosine for completed epochs beyond warmup
        if global_step >= warmup_steps:
            epochs_in_cosine = start_epoch - 1
            for _ in range(epochs_in_cosine):
                cosine_sched.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Scheduler fast-forwarded (step={global_step}) — LR = {current_lr:.4e}")
        print("Optimizer state loaded ✓  (scheduler reconstructed)")

    criterion = nn.L1Loss()

    print(f"\nResuming epoch {start_epoch} → {total_epochs}")
    print(f"Loss weights: E={w_energy}  F={w_force}  |  grad_clip={grad_clip}")
    print(f"Warmup steps: {warmup_steps}  (global_step already at {global_step})")

    # ── SIGTERM handler ───────────────────────────────────────────
    _state = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'run_dir': run_dir, 'best_e_mae': best_e_mae,
        'energy_mean': energy_mean, 'energy_std': energy_std,
        'train_losses': train_losses, 'val_losses': val_losses,
        'val_e_maes': val_e_maes, 'epoch': start_epoch - 1,
        'global_step': global_step, 'wandb_run_id': wandb_run_id,
    }

    def _sigterm_handler(signum, frame):
        ep = _state['epoch']
        gs = _state['global_step']
        path = os.path.join(_state['run_dir'], f'emergency_checkpoint_epoch_{ep:03d}.pt')
        print(f"\n  ⚠ SIGTERM — saving emergency checkpoint: {path}")
        save_checkpoint(path, make_ckpt(
            ep, gs,
            _state['model'], _state['optimizer'], _state['scheduler'],
            _state['best_e_mae'], _state['energy_mean'], _state['energy_std'],
            _state['train_losses'], _state['val_losses'], _state['val_e_maes'],
            wandb_run_id=_state['wandb_run_id'],
        ))
        wandb.log({'emergency_checkpoint_epoch': ep})
        wandb.finish()
        print("  Emergency checkpoint saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── CSV log (append) ──────────────────────────────────────────
    csv_path   = os.path.join(run_dir, 'losses.csv')
    csv_exists = os.path.exists(csv_path)
    csv_file   = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss',
                             'val_e_mae_meV', 'lr', 'nan_batches'])

    run_start = datetime.now()

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs + 1):
        _state['epoch'] = epoch

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
            _state['global_step'] = global_step

            # warmup: step every batch; cosine: step every epoch
            if global_step <= warmup_steps:
                scheduler.step()

            is_nan = (tot != tot)
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
        _state['train_losses'] = train_losses

        if nan_batches > 0:
            print(f"\n  ⚠ {nan_batches} NaN batches skipped this epoch")
            wandb.log({'train/nan_batches_epoch': nan_batches}, step=global_step)

        # Validate
        epoch_val_loss, epoch_e_mae = validate(
            model, val_loader, criterion, energy_std, device
        )
        val_losses.append(epoch_val_loss)
        val_e_maes.append(epoch_e_mae)
        _state['val_losses'] = val_losses
        _state['val_e_maes'] = val_e_maes

        # cosine steps every epoch after warmup
        if global_step > warmup_steps:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']

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

        # Best model
        if epoch_e_mae < best_e_mae:
            best_e_mae = epoch_e_mae
            _state['best_e_mae'] = best_e_mae
            best_path = os.path.join(run_dir, 'best_model_checkpoint.pt')
            save_checkpoint(best_path, make_ckpt(
                epoch, global_step, model, optimizer, scheduler, best_e_mae,
                energy_mean, energy_std, train_losses, val_losses, val_e_maes,
                wandb_run_id=wandb_run_id,
            ))
            print(f"  ✓ Best saved  E MAE={best_e_mae:.2f} meV/atom")
            wandb.log({'val/best_e_mae_meV': best_e_mae}, step=global_step)
            artifact = wandb.Artifact(
                name='best-model', type='model',
                description=f'Best val E MAE = {best_e_mae:.3f} meV/atom at epoch {epoch}',
            )
            artifact.add_file(best_path)
            run.log_artifact(artifact)

        # Periodic checkpoint
        if save_interval and epoch % save_interval == 0:
            ckpt_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            save_checkpoint(ckpt_path, make_ckpt(
                epoch, global_step, model, optimizer, scheduler, best_e_mae,
                energy_mean, energy_std, train_losses, val_losses, val_e_maes,
                wandb_run_id=wandb_run_id,
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
            'resumed_from_epoch':  start_epoch - 1,
        }, f, indent=2)

    wandb.log({
        'test/e_mae_meV':     test_e_mae,
        'test/f_mae_meV':     test_f_mae,
        'run_duration_hours': duration / 3600,
    })
    wandb.finish()

    print(f"\nDone in {duration/3600:.2f}h   Best val E MAE: {best_e_mae:.2f} meV/atom")
    print(f"Results in: {run_dir}")


if __name__ == '__main__':
    main()