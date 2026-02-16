"""Test EquiformerV2 with OC20 data - FIXED VERSION"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your configs and modules
import torch
from configs.OC20.oc20_config import config
from data_loader_oc20 import get_oc20_loaders
from equiformerv2_oc20 import EquiformerV2_OC20

print("=" * 60)
print("TESTING EQUIFORMERV2 WITH OC20 DATA")
print("=" * 60)

# Get one batch
print("\nLoading data...")
train_loader, _ = get_oc20_loaders(
    train_dir=config['train_src'],
    val_dir=config['val_src'],
    batch_size=2,
    num_workers=0,
    max_train_samples=1,
    max_val_samples=1
)

batch = next(iter(train_loader))
print(f"✓ Loaded batch with {len(batch['energy'])} structures")
print(f"  Total atoms: {len(batch['atomic_numbers'])}")

# Create model
print("\nInitializing model...")
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
    num_layers=4,  # Use smaller model for testing
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
)

print(f"✓ Model created with {model.num_params:,} parameters")

# Forward pass
print("\nRunning forward pass...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

# FIXED: Move ALL batch data to device (including cell)
for k in batch:
    batch[k] = batch[k].to(device)

with torch.no_grad():
    energy_pred, forces_pred = model(batch)

print(f"✓ Forward pass successful!")
print(f"  Energy pred shape: {energy_pred.shape}")
print(f"  Forces pred shape: {forces_pred.shape}")
print(f"  Energy target shape: {batch['energy'].shape}")
print(f"  Forces target shape: {batch['forces'].shape}")

# Check loss
import torch.nn as nn
energy_loss = nn.L1Loss()(energy_pred, batch['energy'])
force_loss = nn.L1Loss()(forces_pred, batch['forces'])
print(f"\n  Energy MAE: {energy_loss.item():.4f}")
print(f"  Force MAE: {force_loss.item():.4f}")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Ready to train!")
print("=" * 60)