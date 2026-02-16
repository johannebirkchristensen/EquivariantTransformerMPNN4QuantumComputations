#!/usr/bin/env python3
"""
Training script for EquiformerV2 on OC20 S2EF-2M
Following OCP official implementation structure
"""
import torch
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

import os
import sys
import math
from datetime import datetime
import json
import csv

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.OC20.oc20_config_corrected import config
from data_loader_oc20v2 import get_oc20_loaders, denormalize_energy, denormalize_forces
from equiformerv2_oc20 import EquiformerV2_OC20


# ============================================================================
# HELPER FUNCTIONS (from OCP)
# ============================================================================

def add_weight_decay(model, weight_decay, skip_list=()):
    """
    Add weight decay but skip certain parameters (biases, norms, etc.)
    From OCP implementation
    """
    decay = []
    no_decay = []
    name_no_wd = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        
        # Skip these parameter types from weight decay
        if (name.endswith(".bias") 
            or name.endswith(".affine_weight")
            or name.endswith(".affine_bias") 
            or name.endswith('.mean_shift')
            or 'bias.' in name
            or any(name.endswith(skip_name) for skip_name in skip_list)):
            no_decay.append(param)
            name_no_wd.append(name)
        else:
            decay.append(param)
    
    name_no_wd.sort()
    params = [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]
    return params, name_no_wd


class ExponentialMovingAverage:
    """
    Exponential Moving Average of model parameters
    From OCP implementation
    """
    def __init__(self, parameters, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in parameters:
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, parameters):
        """Update EMA parameters"""
        for name, param in parameters:
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def store(self, parameters):
        """Store current parameters before copying EMA"""
        for name, param in parameters:
            if param.requires_grad:
                self.backup[name] = param.data.clone()
    
    def restore(self, parameters):
        """Restore original parameters"""
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.backup[name])
    
    def copy_to(self, parameters):
        """Copy EMA parameters to model"""
        for name, param in parameters:
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def get_lr_lambda_cosine_warmup(warmup_steps, total_steps, eta_min_ratio=0.0):
    """
    Cosine annealing with linear warmup
    From OCP implementation
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / warmup_steps
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class L2MAELoss(nn.Module):
    """
    L2 loss for energy, MAE for forces
    Used in OCP for S2EF
    """
    def __init__(self):
        super().__init__()
        self.energy_loss = nn.MSELoss()
        self.force_loss = nn.L1Loss()
    
    def forward(self, energy_pred, energy_target, forces_pred, forces_target, 
                energy_coef=1.0, force_coef=1.0):
        """
        Compute combined loss
        """
        e_loss = self.energy_loss(energy_pred, energy_target)
        f_loss = self.force_loss(forces_pred, forces_target)
        return energy_coef * e_loss + force_coef * f_loss


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")
    
    # Create run directory
    run_start_time = datetime.now()
    timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['save_dir'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run directory: {run_dir}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("\n" + "=" * 60)
    print("LOADING OC20 DATA")
    print("=" * 60)
    
    train_loader, val_loader = get_oc20_loaders(
        train_dir=config['train_src'],
        val_dir=config['val_src'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4),
        normalize=config.get('normalize_labels', True),
        max_train_samples=config.get('max_train_samples', None),
        max_val_samples=config.get('max_val_samples', None),
        quick_init=False  # Use exact counting
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {config['batch_size']}")
    
    # ========================================================================
    # MODEL
    # ========================================================================
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
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
    
    print(f"\nModel parameters: {model.num_params:,}")
    print(f"Model size: {model.num_params * 4 / 1e6:.2f} MB")
    
    # ========================================================================
    # OPTIMIZER WITH WEIGHT DECAY EXCLUSION
    # ========================================================================
    print("\n" + "=" * 60)
    print("SETTING UP OPTIMIZER")
    print("=" * 60)
    
    # Get parameters that should not have weight decay
    model_params_no_wd = set()
    if hasattr(model, 'no_weight_decay'):
        model_params_no_wd = model.no_weight_decay()
    
    # Split parameters
    parameters, name_no_wd = add_weight_decay(
        model, 
        config['weight_decay'], 
        model_params_no_wd
    )
    
    print(f"Parameters without weight decay ({len(name_no_wd)}):")
    for name in name_no_wd[:10]:  # Print first 10
        print(f"  {name}")
    if len(name_no_wd) > 10:
        print(f"  ... and {len(name_no_wd) - 10} more")
    
    optimizer = AdamW(
        parameters,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # ========================================================================
    # LEARNING RATE SCHEDULER
    # ========================================================================
    total_epochs = config['epochs']
    steps_per_epoch = len(train_loader)
    warmup_steps = int(config['warmup_epochs'] * steps_per_epoch)
    total_steps = total_epochs * steps_per_epoch
    
    lr_lambda = get_lr_lambda_cosine_warmup(warmup_steps, total_steps, eta_min_ratio=0.0)
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    print(f"\nScheduler: Cosine with linear warmup")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Initial LR: {config['learning_rate']:.2e}")
    
    # ========================================================================
    # EXPONENTIAL MOVING AVERAGE (EMA)
    # ========================================================================
    ema = None
    if config.get('ema_decay', 0) > 0:
        ema = ExponentialMovingAverage(
            model.named_parameters(),
            config['ema_decay']
        )
        print(f"\nUsing EMA with decay: {config['ema_decay']}")
    
    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    criterion = L2MAELoss()
    energy_coef = config['energy_coefficient']
    force_coef = config['force_coefficient']
    
    print(f"\nLoss: {energy_coef} * Energy (MSE) + {force_coef} * Forces (MAE)")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    learning_rates = []
    
    step = 0
    
    for epoch in range(1, total_epochs + 1):
        # ====================================================================
        # TRAINING
        # ====================================================================
        model.train()
        train_loss_sum = 0.0
        train_energy_loss_sum = 0.0
        train_force_loss_sum = 0.0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            energy_pred, forces_pred = model(batch)
            
            # Compute individual losses for logging
            energy_loss = nn.MSELoss()(energy_pred, batch['energy'])
            force_loss = nn.L1Loss()(forces_pred, batch['forces'])
            
            # Combined loss
            loss = energy_coef * energy_loss + force_coef * force_loss
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if config.get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['gradient_clip_norm']
                )
            
            optimizer.step()
            scheduler.step()
            
            # Update EMA
            if ema is not None:
                ema.update(model.named_parameters())
            
            # Accumulate metrics
            bsz = len(batch['energy'])
            train_loss_sum += loss.item() * bsz
            train_energy_loss_sum += energy_loss.item() * bsz
            train_force_loss_sum += force_loss.item() * bsz
            train_samples += bsz
            step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'E': f"{energy_loss.item():.4f}",
                'F': f"{force_loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        epoch_train_loss = train_loss_sum / train_samples
        epoch_train_energy_loss = train_energy_loss_sum / train_samples
        epoch_train_force_loss = train_force_loss_sum / train_samples
        
        train_losses.append(epoch_train_loss)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        # ====================================================================
        # VALIDATION (with EMA if enabled)
        # ====================================================================
        model.eval()
        
        # Store model parameters and copy EMA if using EMA
        if ema is not None:
            ema.store(model.named_parameters())
            ema.copy_to(model.named_parameters())
        
        val_loss_sum = 0.0
        val_energy_loss_sum = 0.0
        val_force_loss_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                
                energy_pred, forces_pred = model(batch)
                
                energy_loss = nn.MSELoss()(energy_pred, batch['energy'])
                force_loss = nn.L1Loss()(forces_pred, batch['forces'])
                loss = energy_coef * energy_loss + force_coef * force_loss
                
                bsz = len(batch['energy'])
                val_loss_sum += loss.item() * bsz
                val_energy_loss_sum += energy_loss.item() * bsz
                val_force_loss_sum += force_loss.item() * bsz
                val_samples += bsz
        
        epoch_val_loss = val_loss_sum / val_samples
        epoch_val_energy_loss = val_energy_loss_sum / val_samples
        epoch_val_force_loss = val_force_loss_sum / val_samples
        
        val_losses.append(epoch_val_loss)
        
        # Restore model parameters if using EMA
        if ema is not None:
            ema.restore(model.named_parameters())
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Train - Total: {epoch_train_loss:.4f}, E: {epoch_train_energy_loss:.4f}, F: {epoch_train_force_loss:.4f}")
        print(f"  Val   - Total: {epoch_val_loss:.4f}, E: {epoch_val_energy_loss:.4f}, F: {epoch_val_force_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # ====================================================================
        # SAVE BEST MODEL (with EMA parameters if using EMA)
        # ====================================================================
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            
            # If using EMA, save EMA parameters
            if ema is not None:
                ema.store(model.named_parameters())
                ema.copy_to(model.named_parameters())
            
            ckpt = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow if ema else None,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            
            torch.save(ckpt, os.path.join(run_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (EMA)" if ema else "  ✓ Saved best model")
            
            # Restore if using EMA
            if ema is not None:
                ema.restore(model.named_parameters())
        
        # Periodic checkpoint
        if config.get('save_interval', 0) > 0 and epoch % config['save_interval'] == 0:
            ckpt = {
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow if ema else None,
                'config': config,
            }
            torch.save(ckpt, os.path.join(run_dir, f'checkpoint_epoch_{epoch:03d}.pt'))
    
    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    metrics = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'total_steps': step,
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


if __name__ == '__main__':
    main()