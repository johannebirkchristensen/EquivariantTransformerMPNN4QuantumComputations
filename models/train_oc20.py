"""
Training script for EquiformerV2 on OC20 S2EF-2M - FIXED VERSION
------------------------------------------------------------------
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

import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from configs.OC20.oc20_config import config
from data_loader_oc20v2 import get_oc20_loaders, denormalize_energy, denormalize_forces
from equiformerv2_oc20 import EquiformerV2_OC20


def main():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print("GPU:", torch.cuda.get_device_name(0))
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.2f} GB")
    
    # Load config
    batch_size = config['batch_size']
    total_epochs = config['epochs']
    save_dir = config['save_dir']
    
    # Create run directory
    run_start_time = datetime.now()
    timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Run directory: {run_dir}")
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING OC20 DATA")
    print("=" * 60)
    
    train_loader, val_loader = get_oc20_loaders(
        train_dir=config['train_src'],
        val_dir=config['val_src'],
        batch_size=batch_size,
        num_workers=config.get('num_workers', 4),
        normalize=config.get('normalize_labels', True),
        max_train_samples=config.get('max_train_samples', None),
        max_val_samples=config.get('max_val_samples', None)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {batch_size}")
    
    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    
    model = EquiformerV2_OC20(
        num_atoms=None,  # Not used
        bond_feat_dim=None,  # Not used
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
    
    print(f"\nModel params: {model.num_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    warmup_epochs = config['warmup_epochs']
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=int(warmup_epochs * len(train_loader))
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int((total_epochs - warmup_epochs) * len(train_loader)),
        eta_min=0.0
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[int(warmup_epochs * len(train_loader))]
    )
    
    # Loss functions
    energy_loss_fn = nn.L1Loss() if config.get('loss_energy', 'mae') == 'mae' else nn.MSELoss()
    force_loss_fn = nn.L1Loss() if config.get('loss_force', 'mae') == 'mae' else nn.MSELoss()
    
    energy_coef = config['energy_coefficient']
    force_coef = config['force_coefficient']
    
    print(f"\nLoss: {energy_coef} * Energy + {force_coef} * Forces")
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_energy_loss_sum = 0.0
        train_force_loss_sum = 0.0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
        for batch in pbar:
            # FIXED: Move all batch data to device (including cell)
            for k in batch:
                batch[k] = batch[k].to(device)
            
            optimizer.zero_grad()
            
            energy_pred, forces_pred = model(batch)
            
            # Compute losses
            energy_loss = energy_loss_fn(energy_pred, batch['energy'])
            force_loss = force_loss_fn(forces_pred, batch['forces'])
            
            loss = energy_coef * energy_loss + force_coef * force_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            optimizer.step()
            scheduler.step()
            
            bsz = len(batch['energy'])
            train_loss_sum += loss.item() * bsz
            train_energy_loss_sum += energy_loss.item() * bsz
            train_force_loss_sum += force_loss.item() * bsz
            train_samples += bsz
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'E': f"{energy_loss.item():.4f}",
                'F': f"{force_loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        epoch_train_loss = train_loss_sum / train_samples
        epoch_train_energy_loss = train_energy_loss_sum / train_samples
        epoch_train_force_loss = train_force_loss_sum / train_samples
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_energy_loss_sum = 0.0
        val_force_loss_sum = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # FIXED: Move all batch data to device (including cell)
                for k in batch:
                    batch[k] = batch[k].to(device)
                
                energy_pred, forces_pred = model(batch)
                
                energy_loss = energy_loss_fn(energy_pred, batch['energy'])
                force_loss = force_loss_fn(forces_pred, batch['forces'])
                
                loss = energy_coef * energy_loss + force_coef * force_loss
                
                bsz = len(batch['energy'])
                val_loss_sum += loss.item() * bsz
                val_energy_loss_sum += energy_loss.item() * bsz
                val_force_loss_sum += force_loss.item() * bsz
                val_samples += bsz
        
        epoch_val_loss = val_loss_sum / val_samples
        epoch_val_energy_loss = val_energy_loss_sum / val_samples
        epoch_val_force_loss = val_force_loss_sum / val_samples
        
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Train - Total: {epoch_train_loss:.4f}, E: {epoch_train_energy_loss:.4f}, F: {epoch_train_force_loss:.4f}")
        print(f"  Val   - Total: {epoch_val_loss:.4f}, E: {epoch_val_energy_loss:.4f}, F: {epoch_val_force_loss:.4f}")
        
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
                'config': config,
            }
            torch.save(ckpt, os.path.join(run_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model")
    
    print("\nTRAINING COMPLETE ✓")


if __name__ == '__main__':
    main()