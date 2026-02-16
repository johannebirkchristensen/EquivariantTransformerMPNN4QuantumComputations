"""
QM9 Configuration for EquiformerV2 - Group 1
============================================
Training on: μ, α, ε_HOMO, ε_LUMO, Δε, C_v

Based on paper Table 13 (page 27) and Section F.2 (page 26)
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
    
    # Architecture - from Table 13
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
    'save_interval': 50,  # Save every 50 epochs
    'save_dir': 'trained_models/QM9',
}

# ================================================================
# CONFIG: μ, α, ε_HOMO, ε_LUMO, Δε, C_v (GROUP 1)
# ================================================================
# Properties: 4=μ, 0=α, 2=ε_HOMO, 3=ε_LUMO, 1=Δε, 5=C_v
# Note: Order in target_indices matches the order in your data loader

config_group1 = base_config.copy()
config_group1.update({
    # Targets
    'num_targets': 6,
    'target_indices': [4, 0, 2, 3, 1, 5],  # μ, α, ε_HOMO, ε_LUMO, Δε, C_v
    
    # Training hyperparameters (from paper Section F.2)
    'batch_size': 64,                  # Paper: 64
    'learning_rate': 5e-4,             # Paper: 5 × 10^-4
    'num_layers': 6,                   # Paper: 6 Transformer blocks
    'weight_decay': 5e-3,              # Paper: 5 × 10^-3
    'alpha_drop': 0.2,                 # Paper: dropout rate 0.2
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,            # Paper: stochastic depth 0.05
    
    # Training precision
    'use_mixed_precision': True,       # Paper: "use mixed precision for training"
})

# ================================================================
# CONFIG: G, H, U, U₀ (ENERGIES) - for comparison
# ================================================================
config_energies = base_config.copy()
config_energies.update({
    'num_targets': 4,
    'target_indices': [6, 7, 9, 10],   # G, H, U, U₀
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 0.0,               # NO weight decay for energies!
    'alpha_drop': 0.0,                 # NO dropout for energies!
    'proj_drop': 0.0,
    'drop_path_rate': 0.0,             # NO stochastic depth for energies!
    'use_mixed_precision': False,      # Single precision
})

# ================================================================
# CONFIG: R² - for comparison
# ================================================================
config_r2 = base_config.copy()
config_r2.update({
    'num_targets': 1,
    'target_indices': [8],             # R² only
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 5e-3,
    'alpha_drop': 0.1,
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,
    'use_mixed_precision': False,      # Single precision
})

# ================================================================
# CONFIG: ZPVE - for comparison
# ================================================================
config_zpve = base_config.copy()
config_zpve.update({
    'num_targets': 1,
    'target_indices': [11],            # ZPVE only
    'batch_size': 48,
    'learning_rate': 1.5e-4,
    'num_layers': 5,
    'weight_decay': 5e-3,
    'alpha_drop': 0.2,
    'proj_drop': 0.0,
    'drop_path_rate': 0.05,
    'use_mixed_precision': False,      # Single precision
})

# ================================================================
# DEFAULT CONFIG - USE GROUP 1
# ================================================================
config = config_group1  # <-- Train on μ, α, ε_HOMO, ε_LUMO, Δε, C_v

# ================================================================
# EXPECTED RESULTS (from paper Table 5, page 10)
# ================================================================
"""
Property         | EquiformerV2 | Equiformer | Units
-----------------|--------------|------------|-------------
μ (dipole)       | 0.010        | 0.011      | Debye
α (polarizability)| 0.050       | 0.046      | Bohr³
ε_HOMO           | 14           | 15         | meV
ε_LUMO           | 13           | 14         | meV
Δε (gap)         | 29           | 30         | meV
C_v (heat cap)   | 0.023        | 0.023      | cal/(mol·K)

Training details (Section F.2, page 26):
- Batch size: 64
- Learning rate: 5e-4
- Epochs: 300
- Weight decay: 5e-3
- Dropout: 0.2 (alpha_drop)
- Stochastic depth: 0.05 (drop_path_rate)
- Transformer blocks: 6
- Mixed precision: Yes
- Training time: ~72 GPU-hours (A6000)
- Model parameters: 11.20M (6 blocks)
"""

# ================================================================
# PROPERTY ORDER IN DATA LOADER
# ================================================================
"""
Index | Property | Units
------|----------|-------------
0     | α        | Bohr³
1     | Δε       | meV
2     | ε_HOMO   | meV
3     | ε_LUMO   | meV
4     | μ        | Debye
5     | C_v      | cal/(mol·K)
6     | G        | meV (kcal/mol → meV conversion)
7     | H        | meV (kcal/mol → meV conversion)
8     | R²       | Bohr²
9     | U        | meV (kcal/mol → meV conversion)
10    | U₀       | meV (kcal/mol → meV conversion)
11    | ZPVE     | meV

For Group 1 (μ, α, ε_HOMO, ε_LUMO, Δε, C_v):
target_indices = [4, 0, 2, 3, 1, 5]

NOTE: Order doesn't matter for training, but keep consistent for evaluation!
"""