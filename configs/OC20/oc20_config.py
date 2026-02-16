"""
OC20 S2EF-2M Configuration for EquiformerV2
============================================
Hyperparameters matching the EquiformerV2 paper Table 7:
"EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations"

All values are taken directly from the S2EF-2M column in Table 7.
"""

config = {
    # ================================================================
    # DATA SETTINGS
    # ================================================================
    'train_src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/data/s2ef/raw/s2ef_train_2M/s2ef_train_2M/',
    'val_src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/data/s2ef/raw/s2ef_val_id_uncompressed/',
    'format': 'extxyz',
    'target_mean': -0.7554450631141663,
    'target_std': 2.887317180633545,
    'grad_target_mean': 0.0,
    'grad_target_std': 2.887317180633545,
    # Dataset settings
    'normalize_labels': True,
    'train_on_free_atoms': True,
    'eval_on_free_atoms': True,
    
    # Number of prediction targets
    'num_targets': 1,  # Energy (forces derived from gradient)
    
    # ================================================================
    # OPTIMIZER SETTINGS (from paper Table 7)
    # ================================================================
    'optimizer': 'AdamW',  # Paper: AdamW
    
    'learning_rate': 2e-4,  # Paper: 2×10⁻⁴ for S2EF-2M
    
    'weight_decay': 1e-3,   # Paper: 1×10⁻³
    
    # ================================================================
    # LEARNING RATE SCHEDULING (from paper Table 7)
    # ================================================================
    'lr_scheduler': 'cosine_with_warmup',  # Cosine annealing with linear warmup.. paper: LambdaLR
    
    'warmup_epochs': 0.1,     # Paper: 0.1 epochs (10% of first epoch)
    
    'epochs': 12,             # Paper: 12 epochs for S2EF-2M
    'loss_energy': 'mae',
    'loss_force': 'l2mae',
    # ================================================================
    # TRAINING SETTINGS (from paper Table 7)
    # ================================================================
    'batch_size': 4,         # Paper: 4, perhaps change to 64 
                              # Reduce to 32 or 16 if OOM
    
    'eval_batch_size': 4,
    
    'num_workers': 8,
    
    # Gradient clipping (from paper Table 7)
    'gradient_clip_norm': 100,  # Paper: 100
    
    # ================================================================
    # REGULARIZATION (from paper Table 7)
    # ================================================================
    'alpha_drop': 0.1,        # Paper: Dropout rate = 0.1
    'proj_drop': 0.1, # paper 0.0
    
    'drop_path_rate': 0.05,   # Paper: Stochastic depth = 0.05
    
    # ================================================================
    # LOSS COEFFICIENTS (from paper Table 7)
    # ================================================================
    'energy_coefficient': 2,   # Paper: λE = 2
    'force_coefficient': 100,  # Paper: λF = 100
    
    # ================================================================
    # MODEL EMA (from paper Table 7)
    # ================================================================
    'ema_decay': 0.999,        # Paper: 0.999
    
    # ================================================================
    # GRAPH CONSTRUCTION (from paper Table 7)
    # ================================================================
    'cutoff_radius': 12.0,     # Paper: 12 Ångström
    
    'max_neighbors': 20,       # Paper: 20
    
    # ================================================================
    # DISTANCE ENCODING (from paper Table 7)
    # ================================================================
    'num_radial_bases': 600,   # Paper: 600 radial basis functions
    
    'distance_function': 'gaussian',
    
    # ================================================================
    # MODEL ARCHITECTURE - CORE (from paper Table 7)
    # ================================================================
    # Maximum degree L_max
    'lmax_list': [6],          # Paper: L_max = 6
    
    # Maximum order M_max
    'mmax_list': [2],          # Paper: M_max = 2
    
    # Number of transformer blocks
    'num_layers': 12,          # Paper: 12 blocks for S2EF-2M
    
    # ================================================================
    # MODEL ARCHITECTURE - DIMENSIONS (from paper Table 7)
    # ================================================================
    # Embedding dimension: (6, 128)
    'sphere_channels': 128,    # Paper: d_embed^(6) = (6, 128)
    
    # Attention hidden dimension: (6, 64)  
    'attn_hidden_channels': 64, # Paper: d_attn_hidden^(6) = (6, 64)
    
    # Number of attention heads
    'num_heads': 8,            # Paper: h = 8
    
    # Attention alpha (for computing attention weights): (0, 64)
    'attn_alpha_channels': 64, # Paper: d_attn_alpha^(0) = (0, 64)
    
    # Attention value dimension: (6, 16)
    'attn_value_channels': 16, # Paper: d_attn_value^(6) = (6, 16)
    
    # FFN hidden dimension: (6, 128)
    'ffn_hidden_channels': 128, # Paper: d_ffn^(6) = (6, 128)
    
    # Edge channels (scalar features): (0, 128)
    'edge_channels': 128,      # Paper: d_edge^(0) = (0, 128)
    
    # ================================================================
    # S2 ACTIVATION GRID (from paper Table 7)
    # ================================================================
    'grid_resolution': 18,     # Paper: R = 18 point samples
    
    # ================================================================
    # OTHER MODEL SETTINGS
    # ================================================================
    'max_num_elements': 90,    # OC20 has many elements
    
    'use_pbc': True,           # OC20: Periodic boundaries for surfaces
    
    'regress_forces': True,    # OC20: Force prediction task
    
    'otf_graph': True,         # Generate graph on-the-fly
    
    'use_atom_edge_embedding': True,
    'share_atom_edge_embedding': False,
    'use_m_share_rad': False,
    
    'norm_type': 'rms_norm_sh', # RMS normalization from paper config:  'layer_norm_sh'    # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']
    
    'attn_activation': 'silu', #scaled_silu??
    'use_s2_act_attn': False,
    'use_attn_renorm': True,
    'ffn_activation': 'silu', # scaled_silu??
    'use_gate_act': False,
    'use_grid_mlp': False,
    'use_sep_s2_act': True,
    
    'weight_init': 'uniform',
    
    # ================================================================
    # LOSS FUNCTION
    # ================================================================
    'loss_function': 'L2',     # MSE for energy, MAE for forces
    
    # ================================================================
    # LOGGING & CHECKPOINTING
    # ================================================================
    'log_interval': 10,
    'save_interval': 1,        # Save every epoch (only 12 epochs)
    'save_dir': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/OC20_S2EF_2M',
}


# ================================================================
# HYPERPARAMETER MAPPING TO PAPER TABLE 7 (S2EF-2M column)
# ================================================================
"""
Paper Table 7 (S2EF-2M)                 Config Key                      Value
-------------------------------------------------------------------------------------------------
Optimizer                               optimizer                       'AdamW'
Learning rate scheduling                lr_scheduler                    'cosine_with_warmup'
Warmup epochs                           warmup_epochs                   0.1
Maximum learning rate                   learning_rate                   2×10⁻⁴
Batch size                              batch_size                      64
Number of epochs                        epochs                          12
Weight decay                            weight_decay                    1×10⁻³
Dropout rate                            alpha_drop, proj_drop           0.1
Stochastic depth                        drop_path_rate                  0.05
Energy coefficient λE                   energy_coefficient              2
Force coefficient λF                    force_coefficient               100
Gradient clipping norm threshold        gradient_clip_norm              100
Model EMA decay                         ema_decay                       0.999
Cutoff radius (Å)                       cutoff_radius                   12.0
Maximum number of neighbors             max_neighbors                   20
Number of radial bases                  num_radial_bases                600
Hidden scalar features                  edge_channels                   (0, 128)
Maximum degree L_max                    lmax_list                       [6]
Maximum order M_max                     mmax_list                       [2]
Number of Transformer blocks            num_layers                      12
Embedding dimension d_embed^(6)         sphere_channels                 128
Attention hidden d_attn_hidden^(6)      attn_hidden_channels            64
Number of attention heads h             num_heads                       8
Attention alpha d_attn_alpha^(0)        attn_alpha_channels             64
Attention value d_attn_value^(6)        attn_value_channels             16
FFN hidden dimension d_ffn^(6)          ffn_hidden_channels             128
Resolution of point samples R           grid_resolution                 18
"""

# ================================================================
# QUICK CONFIG VARIANTS
# ================================================================

# Debugging config (fast iteration, smaller model)
config_debug = config.copy()
config_debug.update({
    'batch_size': 8,
    'epochs': 2,
    'num_layers': 4,           # Reduce from 12 to 4
    'num_workers': 0,
    'save_interval': 1,
})

# Memory-efficient config (if OOM on your GPU)
config_small_batch = config.copy()
config_small_batch.update({
    'batch_size': 16,          # Reduce from 64 to 16
    'eval_batch_size': 16,
    'gradient_clip_norm': 100,
})

# S2EF-All config (if you get the full dataset later)
config_s2ef_all = config.copy()
config_s2ef_all.update({
    'learning_rate': 4e-4,     # Paper: 4×10⁻⁴ for S2EF-All
    'batch_size': 256,         # Paper: 256
    'epochs': 1,               # Paper: 1 epoch
    'warmup_epochs': 0.01,     # Paper: 0.01
    'num_layers': 20,          # Paper: 20 blocks
    'lmax_list': [6],          # Paper: L_max = 6
    'mmax_list': [3],          # Paper: M_max = 3
    'drop_path_rate': 0.1,     # Paper: 0.1
})
