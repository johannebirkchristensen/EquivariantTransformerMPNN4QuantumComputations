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
torch.serialization.add_safe_globals([slice])
import torch.nn as nn
from torch.optim import Adam
from configs.QM9.config import config
from data_loader_qm9 import get_qm9_loaders
from equiformerv2_qm9 import EquiformerV2_QM9

# =======================
# DEVICE
# =======================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

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
train_loader, val_loader, test_loader = get_qm9_loaders(DB_PATH, batch_size=BATCH_SIZE)
print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

# =======================
# INITIALIZE MODEL
# =======================
model_args = {k: config[k] for k in [
    'num_layers', 'sphere_channels', 'attn_hidden_channels',
    'num_heads', 'attn_alpha_channels', 'attn_value_channels',
    'ffn_hidden_channels', 'lmax_list', 'mmax_list', 'num_targets'
]}

model = EquiformerV2_QM9(**model_args).to(DEVICE)
print("Model details:", model)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5
)

# =======================
# TRAINING LOOP
# =======================
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
