"""
QM9 Configuration for EquiformerV2
===================================
Based on paper: "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations"

IMPORTANT: The paper trains SEPARATE MODELS for different property groups!
This file contains configs for each group.
"""

# ================================================================
# BASE CONFIG (shared across all tasks)
# ================================================================
base_config = {
    # Data
    'db_path': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_atomref_corrected.db',
    'max_samples': None,
    'val_split': 0.1,
    'test_split': 0.1,
    
    # Training
    'optimizer': 'AdamW',
    'lr_scheduler': 'cosine_with_warmup',
    'warmup_epochs': 5,
    'epochs': 300,
    'loss_function': 'L1',
    'gradient_clip_norm': 1.0,
    
    # Graph construction
    'cutoff_radius': 5.0,
    'max_neighbors': 500,
    'num_radial_bases': 128,
    'distance_function': 'gaussian',
    
    # Architecture - shared parameters
    'lmax_list': [4],
    'mmax_list': [4],
    'sphere_channels': 96,
    'attn_hidden_channels': 48,
    'num_heads': 4,
    'attn_alpha_channels': 64,
    'attn_value_channels': 24,
    'ffn_hidden_channels': 96,
    'edge_channels': 64,
    'grid_resolution': 18,
    
    # Model settings
    'max_num_elements': 10,
    'use_pbc': False,
    'regress_forces': False,
    'otf_graph': True,
    'use_atom_edge_embedding': True,
    'share_atom_edge_embedding': False,
    'use_m_share_rad': False,
    'norm_type': 'rms_norm_sh',
    'attn_activation': 'scaled_silu',
    'use_s2_act_attn': False,
    'use_attn_renorm': True,
    'ffn_activation': 'scaled_silu',
    'use_gate_act': False,
    'use_grid_mlp': False,
    'use_sep_s2_act': True,
    'weight_init': 'normal',
    
    # Data loading
    'num_workers': 4,
    
    # Logging
    'log_interval': 10,
    'save_interval': 10,
    'save_dir': 'trained_models/QM9',
}

# ================================================================
# CONFIG 1: μ, α, ε_HOMO, ε_LUMO, Δε, C_v
# ================================================================
# Properties: 0=α, 1=Δε, 2=ε_HOMO, 3=ε_LUMO, 4=μ, 5=C_v
config_group1 = base_config.copy()
config_group1.update({
    'num_targets': 6,
    'target_indices': [0, 1, 2, 3, 4, 5],  # α, Δε, ε_HOMO, ε_LUMO, μ, C_v
    'batch_size': 64,
    'learning_rate': 5e-4,
    'num_layers': 6,
    'weight_decay': 5e-3,
    'alpha_drop': 0.2,
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,
})

# ================================================================
# CONFIG 2: R²
# ================================================================
config_r2 = base_config.copy()
config_r2.update({
    'num_targets': 1,
    'target_indices': [8],  # R² only
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 5e-3,
    'alpha_drop': 0.1,
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,
})

# ================================================================
# CONFIG 3: ZPVE
# ================================================================
config_zpve = base_config.copy()
config_zpve.update({
    'num_targets': 1,
    'target_indices': [11],  # ZPVE only
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 5e-3,
    'alpha_drop': 0.2,
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,
})

# ================================================================
# CONFIG 4: G, H, U, U₀ (THERMODYNAMIC ENERGIES)
# ================================================================
# Properties: 6=G, 7=H, 9=U, 10=U₀
config_energies = base_config.copy()
config_energies.update({
    'num_targets': 4,
    'target_indices': [6, 7, 9, 10],  # G, H, U, U₀
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 0.0,      # NO weight decay for energies!
    'alpha_drop': 0.0,        # NO dropout for energies!
    'proj_drop': 0.0,
    'drop_path_rate': 0.0,    # NO stochastic depth for energies!
})

# ================================================================
# CONFIG 5: ALL 12 PROPERTIES (not paper setup, but useful)
# ================================================================
config_all = base_config.copy()
config_all.update({
    'num_targets': 12,
    'target_indices': None,  # All properties
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 0.0,
    'alpha_drop': 0.0,
    'proj_drop': 0.0,
    'drop_path_rate': 0.0,
})

# ================================================================
# DEFAULT CONFIG - USE config_energies FOR G, H, U, U₀
# ================================================================
config = config_energies  # <-- TRAINING FOR THERMODYNAMIC ENERGIES

# ================================================================
# SUMMARY TABLE
# ================================================================
"""
Property Group          | Batch | LR      | Layers | Weight Decay | Dropout | num_targets
------------------------|-------|---------|--------|--------------|---------|------------
μ,α,ε_HOMO,ε_LUMO,Δε,Cν | 64    | 5e-4    | 6      | 5e-3         | 0.2     | 6
R²                      | 48    | 1.5e-4  | 5      | 5e-3         | 0.1     | 1
ZPVE                    | 48    | 1.5e-4  | 5      | 5e-3         | 0.2     | 1
G, H, U, U₀             | 48    | 1.5e-4  | 5      | 0.0          | 0.0     | 4

Key insight: Thermodynamic energies (G,H,U,U₀) need NO regularization!
"""