#!/usr/bin/env python3
"""
Distributed Training for EquiformerV2 on OC20 S2EF-2M
Supports single-GPU and multi-GPU training
"""
import torch
import os
import sys
from datetime import datetime, timedelta
import json
import csv
import argparse

import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.OC20.oc20_config_corrected import config
from data_loader_oc20v2 import OC20Dataset, collate_oc20
from equiformerv2_oc20 import EquiformerV2_OC20


# ============================================================================
# DISTRIBUTED SETUP
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(minutes=30)
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # Single GPU
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if this is the main process (rank 0)"""
    return rank == 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_weight_decay(model, weight_decay, skip_list=()):
    """Add weight decay but skip certain parameters"""
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if (name.endswith(".bias") or 
            name.endswith(".affine_weight") or
            name.endswith(".affine_bias") or
            'bias.' in name):
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


class ExponentialMovingAverage:
    """EMA for model parameters"""
    def __init__(self, parameters, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def store(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def restore(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.backup[name])
    
    def copy_to(self, parameters):
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def get_lr_lambda_cosine_warmup(warmup_steps, total_steps):
    """Cosine annealing with linear warmup"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


# ============================================================================
# TRAINING AND VALIDATION
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, criterion, 
                energy_coef, force_coef, grad_clip, ema, device, epoch, 
                total_epochs, rank, world_size):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_samples = 0
    
    # Progress bar only on main process
    if is_main_process(rank):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    else:
        pbar = train_loader
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # Forward
        energy_pred, forces_pred = model(batch)
        
        # Compute losses
        energy_loss = nn.MSELoss()(energy_pred, batch['energy'])
        force_loss = nn.L1Loss()(forces_pred, batch['forces'])
        loss = energy_coef * energy_loss + force_coef * force_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model.named_parameters())
        
        # Accumulate metrics
        bsz = len(batch['energy'])
        total_loss += loss.item() * bsz
        total_energy_loss += energy_loss.item() * bsz
        total_force_loss += force_loss.item() * bsz
        total_samples += bsz
        
        # Update progress bar
        if is_main_process(rank):
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'E': f"{energy_loss.item():.4f}",
                'F': f"{force_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # Average metrics across GPUs
    if world_size > 1:
        metrics_tensor = torch.tensor(
            [total_loss, total_energy_loss, total_force_loss, total_samples],
            device=device
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_energy_loss, total_force_loss, total_samples = metrics_tensor.tolist()
    
    avg_loss = total_loss / total_samples
    avg_energy_loss = total_energy_loss / total_samples
    avg_force_loss = total_force_loss / total_samples
    
    return avg_loss, avg_energy_loss, avg_force_loss, scheduler.get_last_lr()[0]


@torch.no_grad()
def validate(model, val_loader, criterion, energy_coef, force_coef, 
             device, rank, world_size, use_ema=False, ema=None):
    """Validate the model"""
    model.eval()
    
    # Use EMA if requested
    if use_ema and ema is not None:
        ema.store(model.named_parameters())
        ema.copy_to(model.named_parameters())
    
    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_samples = 0
    
    for batch in val_loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        
        energy_pred, forces_pred = model(batch)
        
        energy_loss = nn.MSELoss()(energy_pred, batch['energy'])
        force_loss = nn.L1Loss()(forces_pred, batch['forces'])
        loss = energy_coef * energy_loss + force_coef * force_loss
        
        bsz = len(batch['energy'])
        total_loss += loss.item() * bsz
        total_energy_loss += energy_loss.item() * bsz
        total_force_loss += force_loss.item() * bsz
        total_samples += bsz
    
    # Average metrics across GPUs
    if world_size > 1:
        metrics_tensor = torch.tensor(
            [total_loss, total_energy_loss, total_force_loss, total_samples],
            device=device
        )
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss, total_energy_loss, total_force_loss, total_samples = metrics_tensor.tolist()
    
    avg_loss = total_loss / total_samples
    avg_energy_loss = total_energy_loss / total_samples
    avg_force_loss = total_force_loss / total_samples
    
    # Restore model if using EMA
    if use_ema and ema is not None:
        ema.restore(model.named_parameters())
    
    return avg_loss, avg_energy_loss, avg_force_loss


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print(f"DISTRIBUTED TRAINING SETUP")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {device}")
    
    # Create run directory (only on main process)
    if is_main_process(rank):
        run_start_time = datetime.now()
        timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(config['save_dir'], timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Run directory: {run_dir}")
    else:
        run_dir = None
    
    # Broadcast run_dir to all processes
    if world_size > 1:
        run_dir_list = [run_dir] if is_main_process(rank) else [None]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
    
    # Create datasets
    train_dataset = OC20Dataset(
        config['train_src'],
        normalize=config.get('normalize_labels', True),
        max_samples=config.get('max_train_samples', None),
        quick_init=False
    )
    
    val_dataset = OC20Dataset(
        config['val_src'],
        normalize=config.get('normalize_labels', True),
        max_samples=config.get('max_val_samples', None),
        quick_init=False
    )
    
    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_oc20,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_oc20,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )
    
    if is_main_process(rank):
        print(f"Train batches per GPU: {len(train_loader)}")
        print(f"Val batches per GPU: {len(val_loader)}")
        print(f"Batch size per GPU: {config['batch_size']}")
        print(f"Effective global batch size: {config['batch_size'] * world_size}")
    
    # ========================================================================
    # MODEL
    # ========================================================================
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("INITIALIZING MODEL")
        print(f"{'='*60}")
    
    model = EquiformerV2_OC20(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=config['num_targets'],
        use_pbc=config['use_pbc'],
        regress_forces=config['regress_forces'],
        otf_graph=config['otf_graph'],
        max_neighbors=config['max_neighbors'],
        max_radius=config['cutoff_radius'],
        max_num_elements=config['max_num_elements'],
        num_layers=config['num_layers'],
        sphere_channels=config['sphere_channels'],
        attn_hidden_channels=config['attn_hidden_channels'],
        num_heads=config['num_heads'],
        attn_alpha_channels=config['attn_alpha_channels'],
        attn_value_channels=config['attn_value_channels'],
        ffn_hidden_channels=config['ffn_hidden_channels'],
        norm_type=config['norm_type'],
        lmax_list=config['lmax_list'],
        mmax_list=config['mmax_list'],
        grid_resolution=config['grid_resolution'],
        edge_channels=config['edge_channels'],
        use_atom_edge_embedding=config['use_atom_edge_embedding'],
        share_atom_edge_embedding=config['share_atom_edge_embedding'],
        use_m_share_rad=config['use_m_share_rad'],
        distance_function=config['distance_function'],
        num_distance_basis=config['num_radial_bases'],
        attn_activation=config['attn_activation'],
        use_s2_act_attn=config['use_s2_act_attn'],
        use_attn_renorm=config['use_attn_renorm'],
        ffn_activation=config['ffn_activation'],
        use_gate_act=config['use_gate_act'],
        use_grid_mlp=config['use_grid_mlp'],
        use_sep_s2_act=config['use_sep_s2_act'],
        alpha_drop=config['alpha_drop'],
        drop_path_rate=config['drop_path_rate'],
        proj_drop=config['proj_drop'],
        weight_init=config['weight_init']
    ).to(device)
    
    if is_main_process(rank):
        print(f"Model parameters: {model.num_params:,}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    model_params_no_wd = set()
    if hasattr(model_without_ddp, 'no_weight_decay'):
        model_params_no_wd = model_without_ddp.no_weight_decay()
    
    parameters = add_weight_decay(model, config['weight_decay'], model_params_no_wd)
    optimizer = AdamW(parameters, lr=config['learning_rate'])
    
    # Scheduler
    total_epochs = config['epochs']
    steps_per_epoch = len(train_loader)
    warmup_steps = int(config['warmup_epochs'] * steps_per_epoch)
    total_steps = total_epochs * steps_per_epoch
    
    lr_lambda = get_lr_lambda_cosine_warmup(warmup_steps, total_steps)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # EMA
    ema = None
    if config.get('ema_decay', 0) > 0:
        ema = ExponentialMovingAverage(model.named_parameters(), config['ema_decay'])
    
    # Loss
    criterion = None  # Using inline losses
    energy_coef = config['energy_coefficient']
    force_coef = config['force_coefficient']
    grad_clip = config.get('gradient_clip_norm', 0)
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    if is_main_process(rank):
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(1, total_epochs + 1):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_e_loss, train_f_loss, lr = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            energy_coef, force_coef, grad_clip, ema, device,
            epoch, total_epochs, rank, world_size
        )
        
        # Validate
        val_loss, val_e_loss, val_f_loss = validate(
            model, val_loader, criterion, energy_coef, force_coef,
            device, rank, world_size, use_ema=(ema is not None), ema=ema
        )
        
        # Log (only main process)
        if is_main_process(rank):
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(lr)
            
            print(f"\nEpoch {epoch}/{total_epochs}")
            print(f"  Train - Total: {train_loss:.4f}, E: {train_e_loss:.4f}, F: {train_f_loss:.4f}")
            print(f"  Val   - Total: {val_loss:.4f}, E: {val_e_loss:.4f}, F: {val_f_loss:.4f}")
            print(f"  LR: {lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Use EMA for best model
                if ema is not None:
                    ema.store(model.named_parameters())
                    ema.copy_to(model.named_parameters())
                
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema_state_dict': ema.shadow if ema else None,
                    'val_loss': val_loss,
                    'config': config,
                }
                
                torch.save(ckpt, os.path.join(run_dir, 'best_model.pt'))
                print(f"  ✓ Saved best model")
                
                if ema is not None:
                    ema.restore(model.named_parameters())
    
    # Save final results (only main process)
    if is_main_process(rank):
        metrics = {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': best_val_loss,
        }
        
        with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(os.path.join(run_dir, 'losses.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
            for i, (tl, vl, lr) in enumerate(zip(train_losses, val_losses, learning_rates), 1):
                writer.writerow([i, tl, vl, lr])
        
        print(f"\n✓ Training complete!")
        print(f"Results saved to: {run_dir}")
    
    cleanup_distributed()


if __name__ == '__main__':
    main()