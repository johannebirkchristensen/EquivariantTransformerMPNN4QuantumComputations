"""
QM9 Configuration for EquiformerV2
===================================
Hyperparameters matching the EquiformerV2 paper:
"EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations"

All values are taken directly from Table in the paper's Appendix.
"""

config = {
    # ================================================================
    # DATA SETTINGS
    # ================================================================
    'db_path': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db',
    
    # For debugging, use a small subset
    'max_samples': None,  # Set to None for full dataset, or e.g. 1000 for debugging
    
    # Dataset splits
    'val_split': 0.1,   # 10% validation
    'test_split': 0.1,  # 10% test
    
    # Number of prediction targets
    'num_targets': 12,  # QM9 has 12 properties (but we can train on just 1 for debugging)
    
    # ================================================================
    # OPTIMIZER SETTINGS (from paper Table)
    # ================================================================
    'optimizer': 'AdamW',  # Paper uses AdamW (not Adam!)
    
    'learning_rate': 5e-4,  # Paper: 1.5e-4 or 5e-4
                            # Start with 5e-4, use 1.5e-4 if overfitting
    
    'weight_decay': 5e-3,   # Paper: 0.0 or 5e-3
                            # Start with 5e-3, use 0.0 if underfitting
    
    # ================================================================
    # LEARNING RATE SCHEDULING (from paper Table)
    # ================================================================
    'lr_scheduler': 'cosine_with_warmup',  # Cosine annealing with linear warmup
    
    'warmup_epochs': 5,     # Paper: 5 epochs of linear warmup
    
    'epochs': 300,          # Paper: 300 epochs total
    
    # ================================================================
    # TRAINING SETTINGS (from paper Table)
    # ================================================================
    'batch_size': 64,       # Paper: 48 or 64
                            # Use 64 if you have 2x A100 40GB
                            # Use 48 if OOM
    
    # Gradient clipping (not in paper table, but standard practice)
    'gradient_clip_norm': 1.0,
    
    # ================================================================
    # REGULARIZATION (from paper Table)
    # ================================================================
    # Dropout rate: Paper tried 0.0, 0.1, 0.2
    'alpha_drop': 0.1,      # Attention dropout
    'proj_drop': 0.0,       # Projection dropout
    
    # Stochastic depth: Paper tried 0.0, 0.05
    'drop_path_rate': 0.05, # Drop path (stochastic depth)
    
    # ================================================================
    # GRAPH CONSTRUCTION (from paper Table)
    # ================================================================
    'cutoff_radius': 5.0,       # Paper: 5.0 Angstrom
    
    'max_neighbors': 500,       # Paper: 500 (even for small QM9 molecules!)
                                # This ensures all neighbors within cutoff are included
    
    # ================================================================
    # DISTANCE ENCODING (from paper Table)
    # ================================================================
    'num_radial_bases': 128,    # Paper: 128 radial basis functions
    
    'distance_function': 'gaussian',
    
    # ================================================================
    # MODEL ARCHITECTURE - CORE (from paper Table)
    # ================================================================
    # Maximum degree L_max
    'lmax_list': [4],           # Paper: L_max = 4
    
    # Maximum order M_max
    'mmax_list': [4],           # Paper: M_max = 4 (NOT 2!)
    
    # Number of transformer blocks
    'num_layers': 6,            # Paper: 5 or 6
                                # Start with 6, try 5 if overfitting
    
    # ================================================================
    # MODEL ARCHITECTURE - DIMENSIONS (from paper Table)
    # ================================================================
    # Embedding dimension: (4, 96)
    # Meaning: 96 channels at lmax=4
    'sphere_channels': 96,      # Paper: d_embed^(4) = (4,96)
    
    # Attention hidden dimension: (4, 48)  
    'attn_hidden_channels': 48, # Paper: d_attn_hidden^(4) = (4,48)
    
    # Number of attention heads
    'num_heads': 4,             # Paper: h = 4
    
    # Attention alpha (for computing attention weights): (0, 64)
    'attn_alpha_channels': 64,  # Paper: d_attn_alpha^(0) = (0,64)
    
    # Attention value dimension: (4, 24)
    'attn_value_channels': 24,  # Paper: d_attn_value^(4) = (4,24)
    
    # FFN hidden dimension: (4, 96)
    'ffn_hidden_channels': 96,  # Paper: d_ffn^(4) = (4,96)
    
    # Edge channels (scalar features): (0, 64)
    'edge_channels': 64,        # Paper: d_edge^(0) = (0,64)
    
    # ================================================================
    # S2 ACTIVATION GRID (from paper Table)
    # ================================================================
    'grid_resolution': 18,      # Paper: R = 18 point samples
    
    # ================================================================
    # OTHER MODEL SETTINGS
    # ================================================================
    'max_num_elements': 10,     # QM9: H, C, N, O, F (5 elements)
                                # Set to 10 for safety
    
    'use_pbc': False,           # QM9: No periodic boundaries
    
    'regress_forces': False,    # QM9: No force labels
    
    'otf_graph': True,          # Generate graph on-the-fly
    
    'use_atom_edge_embedding': True,
    'share_atom_edge_embedding': False,
    'use_m_share_rad': False,
    
    'norm_type': 'rms_norm_sh', # RMS normalization (paper default)
    
    'attn_activation': 'scaled_silu',
    'use_s2_act_attn': False,
    'use_attn_renorm': True,
    'ffn_activation': 'scaled_silu',
    'use_gate_act': False,
    'use_grid_mlp': False,
    'use_sep_s2_act': True,     # Separable S2 activation (paper default)
    
    'weight_init': 'normal',
    
    # ================================================================
    # 3D DENOISING AUGMENTATION (from paper Table)
    # ================================================================
    # NOTE: These are for data augmentation - requires implementation
    # For now, we'll skip this and implement it later if needed
    
    'use_denoising': False,         # Set to True to enable denoising augmentation
    
    # When use_denoising=True:
    'sigma_denoise': 0.02,          # Paper: σ_denoise = 0.02
    'lambda_denoise': 0.1,          # Paper: λ_denoise = 0.1 (loss coefficient)
    'p_denoise': 0.5,               # Paper: p_denoise = 0.5 (50% probability)
    'r_denoise': 0.125,             # Paper: r_denoise = 0.125 or 0.25 (corrupt ratio)
                                     # Fraction of atoms to corrupt per molecule
    
    # ================================================================
    # LOSS FUNCTION
    # ================================================================
    'loss_function': 'L1',          # Paper uses MAE (L1 loss) for QM9
    
    # ================================================================
    # DATA LOADING
    # ================================================================
    'num_workers': 4,               # Number of data loading workers
                                     # Set to 0 for debugging (easier to see errors)
    
    # ================================================================
    # LOGGING & CHECKPOINTING
    # ================================================================
    'log_interval': 10,             # Log every N batches
    'save_interval': 10,            # Save checkpoint every N epochs
    'save_dir': 'trained_models/QM9',
}


# ================================================================
# HYPERPARAMETER MAPPING TO PAPER TABLE
# ================================================================
"""
Paper Table                             Config Key                      Value
-------------------------------------------------------------------------------------------------
Optimizer                               optimizer                       'AdamW'
Learning rate scheduling                lr_scheduler                    'cosine_with_warmup'
Warmup epochs                           warmup_epochs                   5
Maximum learning rate                   learning_rate                   5e-4 (or 1.5e-4)
Batch size                              batch_size                      64 (or 48)
Number of epochs                        epochs                          300
Weight decay                            weight_decay                    5e-3 (or 0.0)
Dropout rate                            alpha_drop, proj_drop           0.1, 0.0 (or 0.0, 0.2)
Stochastic depth                        drop_path_rate                  0.05 (or 0.0)
Cutoff radius (Å)                       cutoff_radius                   5.0
Maximum number of neighbors             max_neighbors                   500
Number of radial bases                  num_radial_bases                128
Hidden scalar feature dimension         edge_channels                   64
Maximum degree L_max                    lmax_list                       [4]
Maximum order M_max                     mmax_list                       [4]
Number of Transformer blocks            num_layers                      6 (or 5)
Embedding dimension d_embed^(4)         sphere_channels                 96
Attention hidden d_attn_hidden^(4)      attn_hidden_channels            48
Number of attention heads h             num_heads                       4
Attention alpha d_attn_alpha^(0)        attn_alpha_channels             64
Attention value d_attn_value^(4)        attn_value_channels             24
FFN hidden dimension d_ffn^(4)          ffn_hidden_channels             96
Resolution of point samples R           grid_resolution                 18
Noise std σ_denoise                     sigma_denoise                   0.02
Denoising coefficient λ_denoise         lambda_denoise                  0.1
Denoising probability p_denoise         p_denoise                       0.5
Corrupt ratio r_denoise                 r_denoise                       0.125 (or 0.25)
"""

# ================================================================
# QUICK CONFIG VARIANTS
# ================================================================

# Debugging config (fast iteration)
config_debug = config.copy()
config_debug.update({
    'max_samples': 1000,
    'batch_size': 16,
    'epochs': 5,
    'num_layers': 4,
    'num_workers': 0,
})

# Conservative config (if overfitting)
config_conservative = config.copy()
config_conservative.update({
    'learning_rate': 1.5e-4,    # Lower LR
    'num_layers': 5,            # Fewer layers
    'alpha_drop': 0.2,          # More dropout
    'drop_path_rate': 0.05,     # Stochastic depth
})

# Aggressive config (if underfitting)
config_aggressive = config.copy()
config_aggressive.update({
    'learning_rate': 5e-4,      # Higher LR
    'weight_decay': 0.0,        # No weight decay
    'num_layers': 6,            # More layers
    'alpha_drop': 0.0,          # No dropout
    'drop_path_rate': 0.0,      # No stochastic depth
})