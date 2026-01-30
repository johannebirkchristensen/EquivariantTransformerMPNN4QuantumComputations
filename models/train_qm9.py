"""
Training script for EquiformerV2_QM9 (QM9 dataset)
---------------------------------------------------
- Standard PyTorch training loop
- Fully uses config file for hyperparameters
- Supports CPU/GPU
- Uses MSE loss for all 12 targets
- Logs train/val metrics each epoch
- Assumes data loader produces batch dict with keys:
    atomic_numbers, pos, batch, natoms, targets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adds project root

# =======================
# IMPORTS
# =======================
from tqdm import tqdm
import yaml
import json
import csv
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW  # Paper uses AdamW, not Adam!
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
torch.serialization.add_safe_globals([slice])
import torch.nn as nn
from torch.optim import Adam
from configs.QM9.config_equiformerV2 import config
from data_loader_qm9_v2 import get_qm9_loaders
from equiformerv2_qm9 import EquiformerV2_QM9

# ================================================================
# DEVICE
# ================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =======================
# LOAD HYPERPARAMETERS
# =======================
DB_PATH = config.get('db_path', 'qm9.db')
BATCH_SIZE = config.get('batch_size', 16)
LR = config.get('learning_rate', 1e-3)
EPOCHS = config.get('epochs', 50)

# =======================
# LOAD DATA
# =======================


# ================================================================
# LOAD DATA
# ================================================================
print("\n" + "="*60)
print("LOADING QM9 DATASET")
print("="*60)

train_loader, val_loader, test_loader = get_qm9_loaders(
    db_path=config['db_path'],
    batch_size=config['batch_size'],
    val_split=config['val_split'],
    test_split=config['test_split'],
    max_samples=config['max_samples'],  # None for full dataset
    num_workers=config['num_workers']
)

print(f"Train: {len(train_loader.dataset)} samples")
print(f"Val:   {len(val_loader.dataset)} samples")
print(f"Test:  {len(test_loader.dataset)} samples")
print(f"Batch size: {config['batch_size']}")



#_________________________OLD CODE___________________________
#print("Loading QM9 dataset...")
#train_loader, val_loader, test_loader = get_qm9_loaders(DB_PATH, batch_size=BATCH_SIZE)
#print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
#print("QM9 dataset loaded ! ")
# =======================
# INITIALIZE MODEL
# =======================






# ================================================================
# INITIALIZE MODEL
# ================================================================
print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

# Extract model args from config
model_args = {
    'num_targets': config['num_targets'],
    'use_pbc': config['use_pbc'],
    'regress_forces': config['regress_forces'],
    'otf_graph': config['otf_graph'],
    'max_neighbors': config['max_neighbors'],
    'max_radius': config['cutoff_radius'],
    'max_num_elements': config['max_num_elements'],
    'num_layers': config['num_layers'],
    'sphere_channels': config['sphere_channels'],
    'attn_hidden_channels': config['attn_hidden_channels'],
    'num_heads': config['num_heads'],
    'attn_alpha_channels': config['attn_alpha_channels'],
    'attn_value_channels': config['attn_value_channels'],
    'ffn_hidden_channels': config['ffn_hidden_channels'],
    'norm_type': config['norm_type'],
    'lmax_list': config['lmax_list'],
    'mmax_list': config['mmax_list'],
    'grid_resolution': config['grid_resolution'],
    'edge_channels': config['edge_channels'],
    'use_atom_edge_embedding': config['use_atom_edge_embedding'],
    'share_atom_edge_embedding': config['share_atom_edge_embedding'],
    'use_m_share_rad': config['use_m_share_rad'],
    'distance_function': config['distance_function'],
    'num_distance_basis': config['num_radial_bases'],
    'attn_activation': config['attn_activation'],
    'use_s2_act_attn': config['use_s2_act_attn'],
    'use_attn_renorm': config['use_attn_renorm'],
    'ffn_activation': config['ffn_activation'],
    'use_gate_act': config['use_gate_act'],
    'use_grid_mlp': config['use_grid_mlp'],
    'use_sep_s2_act': config['use_sep_s2_act'],
    'alpha_drop': config['alpha_drop'],
    'drop_path_rate': config['drop_path_rate'],
    'proj_drop': config['proj_drop'],
    'weight_init': config['weight_init'],
}

model = EquiformerV2_QM9(**model_args).to(DEVICE)

print(f"Model parameters: {model.num_params:,}")
print(f"Model size: {model.num_params * 4 / 1e6:.2f} MB (float32)")
#_________________________OLD CODE___________________________
#print("Initializing EquiformerV2_QM9 model...")
#model_args = {k: config[k] for k in [
#    'num_layers', 'sphere_channels', 'attn_hidden_channels',
#    'num_heads', 'attn_alpha_channels', 'attn_value_channels',
#    'ffn_hidden_channels', 'lmax_list', 'mmax_list', 'num_targets'
#]}

#model = EquiformerV2_QM9(**model_args).to(DEVICE)
#print("Model initialized! ")
#print("Model details:", model)

#criterion = nn.MSELoss()
#criterion = nn.L1Loss()   # MAE ( They use MAE in equiformer paper for QM9 and OC20 so let's do the same )

#optimizer = Adam(model.parameters(), lr=LR)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, factor=0.5, patience=5
#)


# ================================================================
# OPTIMIZER & SCHEDULER
# ================================================================
print("\n" + "="*60)
print("SETTING UP OPTIMIZER & SCHEDULER")
print("="*60)

# AdamW optimizer (paper uses AdamW, not Adam!)
optimizer = AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

print(f"Optimizer: AdamW")
print(f"Learning rate: {config['learning_rate']}")
print(f"Weight decay: {config['weight_decay']}")

# Cosine annealing with linear warmup (paper's approach)
warmup_epochs = config['warmup_epochs']
total_epochs = config['epochs']

# Warmup: linear increase from 0 to max_lr
warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,      # Start at 1% of max LR
    end_factor=1.0,         # End at 100% of max LR (= learning_rate)
    total_iters=warmup_epochs
)

# Cosine annealing: smooth decay from max_lr to 0
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_epochs - warmup_epochs,  # Remaining epochs after warmup
    eta_min=0.0  # Minimum LR at end
)

# Combine: warmup for first 5 epochs, then cosine
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)

print(f"Scheduler: Cosine with {warmup_epochs} epoch linear warmup")
print(f"Total epochs: {total_epochs}")
# ================================================================
# LOSS FUNCTION
# ================================================================
if config['loss_function'] == 'L1':
    criterion = nn.L1Loss()  # MAE (paper uses this for QM9)
    print(f"Loss: L1 (MAE)")
elif config['loss_function'] == 'MSE':
    criterion = nn.MSELoss()
    print(f"Loss: MSE")
else:
    raise ValueError(f"Unknown loss: {config['loss_function']}")

# ================================================================
# TRAINING LOOP
# ================================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

train_losses = []
val_losses = []
learning_rates = []

best_val_loss = float('inf')
run_start_time = datetime.now()
for epoch in range(1, total_epochs + 1):
    # ============================================================
    # TRAINING
    # ============================================================
    model.train()
    train_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch)  # [batch_size, num_targets]
        
        # Compute loss
        loss = criterion(predictions, batch['targets'])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (helps stability)
        if 'gradient_clip_norm' in config and config['gradient_clip_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['gradient_clip_norm']
            )
        
        optimizer.step()
        
        # Accumulate loss
        batch_size_actual = len(batch['targets'])
        train_loss += loss.item() * batch_size_actual
        num_samples += batch_size_actual
        
        # Update progress bar
        avg_loss = train_loss / num_samples
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg": f"{avg_loss:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    train_loss /= len(train_loader.dataset)
    
    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            
            predictions = model(batch)
            loss = criterion(predictions, batch['targets'])
            
            val_loss += loss.item() * len(batch['targets'])
    
    val_loss /= len(val_loader.dataset)
    
    # ============================================================
    # LEARNING RATE SCHEDULING
    # ============================================================
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # ============================================================
    # LOGGING
    # ============================================================
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    learning_rates.append(current_lr)
    
    print(f"\nEpoch {epoch:03d}/{total_epochs:03d}")
    print(f"  Train Loss: {train_loss:.6f}")
    print(f"  Val Loss:   {val_loss:.6f}")
    print(f"  LR:         {current_lr:.2e}")
    
    # ============================================================
    # SAVE BEST MODEL
    # ============================================================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        # Create save directory if needed
        os.makedirs(config['save_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        
        torch.save(
            checkpoint,
            os.path.join(config['save_dir'], 'best_model_checkpoint.pt')
        )
        
        print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
    
    # ============================================================
    # PERIODIC CHECKPOINTS
    # ============================================================
    if 'save_interval' in config and epoch % config['save_interval'] == 0:
        checkpoint_path = os.path.join(
            config['save_dir'],
            f'checkpoint_epoch_{epoch:03d}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Saved checkpoint: {checkpoint_path}")

# ================================================================
# TESTING
# ================================================================
print("\n" + "="*60)
print("TESTING BEST MODEL")
print("="*60)

# Load best model
best_checkpoint = torch.load(
    os.path.join(config['save_dir'], 'best_model_checkpoint.pt')
)
model.load_state_dict(best_checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {best_checkpoint['epoch']}")

model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        
        predictions = model(batch)
        loss = criterion(predictions, batch['targets'])
        
        test_loss += loss.item() * len(batch['targets'])

test_loss /= len(test_loader.dataset)

print(f"\nTest Loss: {test_loss:.6f}")

# ================================================================
# SAVE FINAL RESULTS
# ================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

run_end_time = datetime.now()
run_duration = (run_end_time - run_start_time).total_seconds()

# Create unique run folder
timestamp = run_end_time.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(config['save_dir'], f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# Save final config with results
config_final = config.copy()
config_final.update({
    'final_train_loss': float(train_losses[-1]),
    'final_val_loss': float(val_losses[-1]),
    'best_val_loss': float(best_val_loss),
    'final_test_loss': float(test_loss),
    'best_epoch': int(best_checkpoint['epoch']),
    'run_start_time': run_start_time.isoformat(),
    'run_end_time': run_end_time.isoformat(),
    'run_duration_seconds': run_duration,
    'model_parameters': model.num_params,
})

# Save config
with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config_final, f, default_flow_style=False, sort_keys=False)

# Copy best model
import shutil
shutil.copy(
    os.path.join(config['save_dir'], 'best_model_checkpoint.pt'),
    os.path.join(run_dir, 'best_model.pt')
)

# Save losses
with open(os.path.join(run_dir, 'losses.json'), 'w') as f:
    json.dump({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'test_loss': test_loss,
        'best_val_loss': best_val_loss,
    }, f, indent=2)

# Save losses as CSV
with open(os.path.join(run_dir, 'losses.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
    for i, (tl, vl, lr) in enumerate(zip(train_losses, val_losses, learning_rates), 1):
        writer.writerow([i, tl, vl, lr])

print(f"\nResults saved to: {run_dir}")
print(f"Best model saved: {os.path.join(run_dir, 'best_model.pt')}")
print(f"Config saved: {os.path.join(run_dir, 'config.yaml')}")
print(f"Training duration: {run_duration/3600:.2f} hours")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)


#_____________________OLD CODE___________________________
"""

# =======================
# TRAINING LOOP
# =======================
print("Starting training...")
train_losses = []
val_losses = []

run_start_time = datetime.now()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    num_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")

    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['targets'])
        loss.backward()
        optimizer.step()

        bs = len(batch['targets'])
        train_loss += loss.item() * bs
        num_samples += bs

        avg_loss = train_loss / num_samples
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"})

    train_loss /= len(train_loader.dataset)

    # =======================
    # VALIDATION
    # =======================
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            outputs = model(batch)
            loss = criterion(outputs, batch['targets'])
            val_loss += loss.item() * len(batch['targets'])

    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# =======================
# TESTING
# =======================
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        for k in batch:
            batch[k] = batch[k].to(DEVICE)
        outputs = model(batch)
        loss = criterion(outputs, batch['targets'])
        test_loss += loss.item() * len(batch['targets'])

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.6f}")

# =======================
# CREATE SAVE FOLDER (ONLY NOW)
# =======================
run_end_time = datetime.now()
run_name = run_end_time.strftime("run_%Y%m%d_%H%M%S")

save_dir = os.path.join("trained_models", "QM9", run_name)
os.makedirs(save_dir, exist_ok=True)

print(f"Saving run to: {save_dir}")

# =======================
# SAVE MODEL
# =======================
model_path = os.path.join(save_dir, "equiformer_v2_qm9.pt")
torch.save(model.state_dict(), model_path)

# =======================
# SAVE CONFIG
# =======================
config_to_save = config.copy()
config_to_save['final_train_loss'] = float(train_losses[-1])
config_to_save['final_val_loss'] = float(val_losses[-1])
config_to_save['final_test_loss'] = float(test_loss)
config_to_save['run_start_time'] = run_start_time.isoformat()
config_to_save['run_end_time'] = run_end_time.isoformat()

config_path = os.path.join(save_dir, "config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config_to_save, f)

# =======================
# SAVE LOSSES (JSON)
# =======================
loss_log = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "test_loss": test_loss
}

with open(os.path.join(save_dir, "losses.json"), "w") as f:
    json.dump(loss_log, f, indent=2)

# =======================
# SAVE LOSSES (CSV)
# =======================
with open(os.path.join(save_dir, "losses.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    for i, (tr, va) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([i, tr, va])

print("====================================")
print("Training complete.")
print(f"Model saved to:  {model_path}")
print(f"Config saved to: {config_path}")
print(f"Losses saved to: {save_dir}")
print("====================================")
"""