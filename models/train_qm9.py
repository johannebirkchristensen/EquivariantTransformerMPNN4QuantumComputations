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

# =======================
# IMPORTS
# =======================
import os
import yaml
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from configs.QM9 import config  # your YAML or Python config
from data_loader_qm9 import get_qm9_loaders
from equiformer_v2_qm9 import EquiformerV2_QM9  # your model

# =======================
# DEVICE
# =======================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =======================
# LOAD HYPERPARAMETERS FROM CONFIG
# =======================
DB_PATH = config.get('db_path', 'qm9.db')  # path to QM9 ASE database
BATCH_SIZE = config.get('batch_size', 16)
LR = config.get('learning_rate', 1e-3)
EPOCHS = config.get('epochs', 50)

# =======================
# LOAD DATA
# =======================
# Returns PyTorch DataLoader objects for training, validation, testing
train_loader, val_loader, test_loader = get_qm9_loaders(DB_PATH, batch_size=BATCH_SIZE)

# =======================
# INITIALIZE MODEL, LOSS, OPTIMIZER
# =======================
# Unpack config directly into the model constructor
model = EquiformerV2_QM9(**config).to(DEVICE)

# Mean squared error loss for QM9 targets
criterion = nn.MSELoss()

# Adam optimizer
optimizer = Adam(model.parameters(), lr=LR)

# Optional learning rate scheduler (reduce LR on plateau)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=5, verbose=True
)

# =======================
# TRAINING LOOP
# =======================
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        # Move all batch tensors to device
        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch)  # Shape: [batch_size, 12]

        loss = criterion(outputs, batch['targets'])
        loss.backward()
        optimizer.step()

        # Multiply by batch size to get sum for averaging later
        train_loss += loss.item() * len(batch['targets'])

    # Average over dataset
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

    # Scheduler step (ReduceLROnPlateau requires metric)
    scheduler.step(val_loss)

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
# SAVE MODEL & CONFIG
# =======================


# Folder path: create a unique folder for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("trained_models", "QM9", f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

# --- Save model state dict ---
model_path = os.path.join(save_dir, "equiformer_v2_qm9.pt")
torch.save(model.state_dict(), model_path)
print(f"Trained model saved at: {model_path}")

# --- Add performance to config ---
config_to_save = config.copy()
config_to_save['final_train_loss'] = train_loss
config_to_save['final_val_loss'] = val_loss
config_to_save['final_test_loss'] = test_loss

# --- Save config as YAML ---
config_path = os.path.join(save_dir, "config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config_to_save, f)
print(f"Config saved at: {config_path}")


"""
Reload the model by 
---------------------------------------------------
import yaml
import torch
from equiformer_v2_qm9 import EquiformerV2_QM9

# Load saved config
with open("trained_models/QM9/run_20260129_123456/config.yaml") as f:
    cfg = yaml.safe_load(f)

model = EquiformerV2_QM9(**cfg)
model.load_state_dict(torch.load("trained_models/QM9/run_20260129_123456/equiformer_v2_qm9.pt"))
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

"""