#!/usr/bin/env python3
"""
Training script for EquiformerV2_QM9 (QM9 dataset)
---------------------------------------------------
- Standard PyTorch training loop
- Fully uses config file for hyperparameters
- Supports CPU/GPU
- Uses L1 (MAE) or MSE depending on config
- Supports training on specific target properties (e.g., only G, H, U, U₀)
- Supports resuming from checkpoint via config OR command-line argument
"""
import torch
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass
import os
import sys
from datetime import datetime
import json
import csv
import shutil
import argparse

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optimizer / scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your configs and modules
from configs.QM9.config_equiformerV2_mu_alpha_homo_lumo_osv import config
from data_loader_qm9_v4 import get_qm9_loaders
from equiformerv2_qm9 import EquiformerV2_QM9


def extract_predictions(out):
    """Extract predictions from model output"""
    if isinstance(out, torch.Tensor):
        preds = out
    elif isinstance(out, dict):
        for key in ('preds', 'predictions', 'energy', 'output'):
            if key in out and isinstance(out[key], torch.Tensor):
                preds = out[key]
                break
        else:
            preds = None
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    preds = v
                    break
            if preds is None:
                raise RuntimeError("Model returned a dict but no tensor found inside it.")
    else:
        raise RuntimeError("Model returned unsupported type: %s" % type(out))

    if not isinstance(preds, torch.Tensor):
        raise RuntimeError("Extracted predictions are not a tensor.")

    return preds


def save_checkpoint(path, checkpoint):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def main():
    # ---- Parse command-line arguments ----
    parser = argparse.ArgumentParser(description='Train EquiformerV2 on QM9')
    parser.add_argument('--checkpoint', '--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., trained_models/QM9/20240210_143022/last_checkpoint.pt)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total number of epochs (overrides config file)')
    args = parser.parse_args()

    # ---- device ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")

    # ---- load hyperparameters from config ----
    db_path = config.get('db_path')
    batch_size = config.get('batch_size', 16)
    total_epochs = config.get('epochs', 50)
    base_save_dir = config.get('save_dir', 'trained_models/QM9')
    num_targets = config.get('num_targets', 12)
    target_indices = config.get('target_indices', None)
    
    # Command-line argument overrides config file
    resume_from = args.checkpoint if args.checkpoint else config.get('resume_from', None)
    if args.epochs is not None:
        total_epochs = args.epochs
        print(f"Overriding epochs from command line: {total_epochs}")

    # ---- create run-specific save directory OR use existing one ----
    run_start_time = datetime.now()
    start_epoch = 1
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    
    # Check if we're resuming from a checkpoint
    if resume_from and os.path.exists(resume_from):
        print("\n" + "=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        print(f"Loading checkpoint: {resume_from}")
        
        checkpoint = torch.load(resume_from, map_location=device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # Use the same directory as the checkpoint
        save_dir = os.path.dirname(resume_from)
        print(f"Continuing training in directory: {save_dir}")
        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.6f}")
        
        # Load existing losses if available
        losses_file = os.path.join(save_dir, 'losses.csv')
        if os.path.exists(losses_file):
            with open(losses_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    train_losses.append(float(row['train_loss']))
                    val_losses.append(float(row['val_loss']))
                    learning_rates.append(float(row['lr']))
            print(f"Loaded {len(train_losses)} previous epochs from losses.csv")
    elif resume_from:
        print(f"\nWARNING: Checkpoint path specified but not found: {resume_from}")
        print("Starting fresh training instead...\n")
        checkpoint = None
        timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        save_dir = run_dir
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    else:
        # Create new run directory
        timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_save_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        save_dir = run_dir
        
        # save the config used for this run
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Run directory: {save_dir}")
        checkpoint = None

    # Print which targets we're training on
    if target_indices is not None:
        property_names = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                          'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']
        selected_properties = [property_names[i] for i in target_indices]
        print(f"\nTraining on specific targets: {selected_properties}")
        print(f"Target indices: {target_indices}")
    else:
        print("\nTraining on all 12 targets")

    # ---- load data ----
    print("\n" + "=" * 60)
    print("LOADING QM9 DATA")
    print("=" * 60)
    train_loader, val_loader, test_loader = get_qm9_loaders(
        db_path=db_path,
        batch_size=batch_size,
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        max_samples=config.get('max_samples', None),
        num_workers=config.get('num_workers', 0),
        shuffle=True,
        target_indices=target_indices
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val   samples: {len(val_loader.dataset)}")
    print(f"Test  samples: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")

    # ---- model init ----
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    model_args = {
        'num_targets': config['num_targets'],
        'use_pbc': config.get('use_pbc', False),
        'regress_forces': config.get('regress_forces', False),
        'otf_graph': config.get('otf_graph', True),
        'max_neighbors': config.get('max_neighbors', 500),
        'max_radius': config.get('cutoff_radius', 5.0),
        'max_num_elements': config.get('max_num_elements', 10),
        'num_layers': config.get('num_layers', 5),
        'sphere_channels': config.get('sphere_channels', 96),
        'attn_hidden_channels': config.get('attn_hidden_channels', 48),
        'num_heads': config.get('num_heads', 4),
        'attn_alpha_channels': config.get('attn_alpha_channels', 64),
        'attn_value_channels': config.get('attn_value_channels', 24),
        'ffn_hidden_channels': config.get('ffn_hidden_channels', 96),
        'norm_type': config.get('norm_type', 'rms_norm_sh'),
        'lmax_list': config.get('lmax_list', [4]),
        'mmax_list': config.get('mmax_list', [4]),
        'grid_resolution': config.get('grid_resolution', 18),
        'edge_channels': config.get('edge_channels', 64),
        'use_atom_edge_embedding': config.get('use_atom_edge_embedding', True),
        'share_atom_edge_embedding': config.get('share_atom_edge_embedding', False),
        'use_m_share_rad': config.get('use_m_share_rad', False),
        'distance_function': config.get('distance_function', 'gaussian'),
        'num_distance_basis': config.get('num_radial_bases', 128),
        'attn_activation': config.get('attn_activation', 'scaled_silu'),
        'use_s2_act_attn': config.get('use_s2_act_attn', False),
        'use_attn_renorm': config.get('use_attn_renorm', True),
        'ffn_activation': config.get('ffn_activation', 'scaled_silu'),
        'use_gate_act': config.get('use_gate_act', False),
        'use_grid_mlp': config.get('use_grid_mlp', False),
        'use_sep_s2_act': config.get('use_sep_s2_act', True),
        'alpha_drop': config.get('alpha_drop', 0.0),
        'drop_path_rate': config.get('drop_path_rate', 0.0),
        'proj_drop': config.get('proj_drop', 0.0),
        'weight_init': config.get('weight_init', 'normal'),
    }
    
    print("\nModel Configuration:")
    for k, v in model_args.items():
        print(f"  {k}: {v}")

    model = EquiformerV2_QM9(**model_args).to(device)
    print(f"\nModel params: {model.num_params:,}")
    print(f"Model size (approx MB): {model.num_params * 4 / 1e6:.2f}")

    # ---- optimizer & scheduler ----
    print("\n" + "=" * 60)
    print("SETTING UP OPTIMIZER & SCHEDULER")
    print("=" * 60)
    optimizer = AdamW(
        model.parameters(), 
        lr=config.get('learning_rate', 1.5e-4),
        weight_decay=config.get('weight_decay', 0.0)
    )

    warmup_epochs = config.get('warmup_epochs', 5)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=0.0
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Load checkpoint states if resuming
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Loaded model, optimizer, and scheduler states from checkpoint")

    print(f"Optimizer: AdamW")
    print(f"  Learning rate: {config.get('learning_rate', 1.5e-4)}")
    print(f"  Weight decay: {config.get('weight_decay', 0.0)}")
    print(f"  Alpha drop: {config.get('alpha_drop', 0.0)}")
    print(f"  Drop path rate: {config.get('drop_path_rate', 0.0)}")
    print(f"Scheduler: linear warmup {warmup_epochs} epochs -> cosine annealing")

    # ---- loss function ----
    loss_name = config.get('loss_function', 'L1')
    if loss_name == 'L1':
        criterion = nn.L1Loss()
        print("Using L1 (MAE) loss")
    elif loss_name == 'MSE':
        criterion = nn.MSELoss()
        print("Using MSE loss")
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    grad_clip = config.get('gradient_clip_norm', 1.0)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    if checkpoint is not None:
        print(f"Continuing from epoch {start_epoch} to {total_epochs}")
    else:
        print(f"Training from epoch 1 to {total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        num_batches = len(train_loader)
        print_every = max(1, num_batches // 100)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            for k in batch:
                batch[k] = batch[k].to(device)

            optimizer.zero_grad()
            out = model(batch)
            preds = extract_predictions(out)

            if preds.dim() == 1:
                B = len(batch['targets'])
                preds = preds.view(B, -1)
            if preds.dim() != 2 or preds.shape[1] != num_targets:
                raise RuntimeError(f"Model predictions must be [B, {num_targets}], got {preds.shape}")

            loss = criterion(preds, batch['targets'])
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bsz = len(batch['targets'])
            train_loss_sum += loss.item() * bsz
            train_samples += bsz

            avg_loss = train_loss_sum / train_samples

            if (batch_idx % print_every == 0) or (batch_idx == num_batches - 1):
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "avg": f"{avg_loss:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                out = model(batch)
                preds = extract_predictions(out)
                if preds.dim() == 1:
                    preds = preds.view(len(batch['targets']), -1)
                if preds.dim() != 2 or preds.shape[1] != num_targets:
                    raise RuntimeError(f"Model predictions must be [B, {num_targets}], got {preds.shape}")

                loss = criterion(preds, batch['targets'])
                val_loss_sum += loss.item() * len(batch['targets'])

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Logging
        print(f"\nEpoch {epoch:03d}/{total_epochs:03d}")
        print(f"  Train Loss: {epoch_train_loss:.6f}")
        print(f"  Val   Loss: {epoch_val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            best_path = os.path.join(save_dir, 'best_model_checkpoint.pt')
            save_checkpoint(best_path, ckpt)
            print(f"  ✓ Saved best model to: {best_path}")

        # Periodic checkpoint (always save last epoch as well for resuming)
        save_interval = config.get('save_interval', 0)
        if (save_interval and epoch % save_interval == 0) or epoch == total_epochs:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            if epoch == total_epochs:
                ckpt_path = os.path.join(save_dir, 'last_checkpoint.pt')
            else:
                ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt_path, ckpt)
            print(f"  ✓ Saved checkpoint: {ckpt_path}")

    # -------------------------------
    # TEST using best model
    # -------------------------------
    print("\n" + "=" * 60)
    print("TESTING BEST MODEL")
    print("=" * 60)

    best_checkpoint_path = os.path.join(save_dir, 'best_model_checkpoint.pt')
    ckpt = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best model from epoch {ckpt.get('epoch', 'unknown')}")

    model.eval()
    test_loss_sum = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            for k in batch:
                batch[k] = batch[k].to(device)
            out = model(batch)
            preds = extract_predictions(out)
            if preds.dim() == 1:
                preds = preds.view(len(batch['targets']), -1)
            if preds.dim() != 2 or preds.shape[1] != num_targets:
                raise RuntimeError(f"Model predictions must be [B, {num_targets}], got {preds.shape}")

            loss = criterion(preds, batch['targets'])
            test_loss_sum += loss.item() * len(batch['targets'])

            all_preds.append(preds.detach().cpu())
            all_targets.append(batch['targets'].detach().cpu())

    test_loss = test_loss_sum / len(test_loader.dataset)
    print(f"\nTest Loss: {test_loss:.6f}")

    # Concatenate and save arrays
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
    else:
        all_preds = None
        all_targets = None

    # Save run artifacts
    run_end_time = datetime.now()
    run_duration = (run_end_time - run_start_time).total_seconds()

    try:
        shutil.copy(best_checkpoint_path, os.path.join(save_dir, 'best_model.pt'))
    except Exception:
        pass

    metrics = {
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'run_start_time': run_start_time.isoformat(),
        'run_end_time': run_end_time.isoformat(),
        'run_duration_seconds': run_duration,
        'model_parameters': model.num_params,
        'target_indices': target_indices,
        'resumed_from_epoch': start_epoch - 1 if checkpoint is not None else 0,
    }
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    if all_preds is not None:
        import numpy as np
        np.save(os.path.join(save_dir, 'predictions.npy'), all_preds)
        np.save(os.path.join(save_dir, 'targets.npy'), all_targets)

    with open(os.path.join(save_dir, 'losses.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
        for i, (tl, vl, lr) in enumerate(zip(train_losses, val_losses, learning_rates), 1):
            writer.writerow([i, tl, vl, lr])

    print(f"\nResults saved to: {save_dir}")
    print("TRAINING COMPLETE")


if __name__ == '__main__':
    main()