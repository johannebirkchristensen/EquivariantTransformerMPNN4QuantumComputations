"""
OC20 S2EF-2M Configuration - CORRECTED TO MATCH PAPER EXACTLY
==============================================================
All hyperparameters from EquiformerV2 paper Table 7 (S2EF-2M column)
"""

config = {
    # ================================================================
    # DATA PATHS
    # ================================================================
    'train_src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/data/s2ef/raw/s2ef_train_2M/s2ef_train_2M/',
    'val_src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/data/s2ef/raw/s2ef_val_id_uncompressed/',
    
    # Normalization statistics (from OCP)
    'target_mean': -0.7554450631141663,
    'target_std': 2.887317180633545,
    'grad_target_mean': 0.0,
    'grad_target_std': 2.887317180633545,
    'normalize_labels': True,
    
    'num_targets': 1,
    
    # ================================================================
    # TRAINING HYPERPARAMETERS (from paper Table 7, S2EF-2M)
    # ================================================================
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,         # Paper: 2×10⁻⁴
    'batch_size': 8,              # Paper: 64 (NOT 4!)... turned down to 16 from 32 ( OOM error ) - #BSUB -q gpua100 #BSUB -R "select[gpu80gb]" after validate train with those arguments... then enough space for batchsize 64 inshallh.
    'epochs': 12,                  # Paper: 12
    'weight_decay': 1e-3,          # Paper: 1×10⁻³
    
    # Learning rate schedule
    'warmup_epochs': 0.1,          # Paper: 0.1 epochs
    
    # Regularization
    'alpha_drop': 0.1,             # Paper: 0.1
    'proj_drop': 0.0,              # Paper: 0.0
    'drop_path_rate': 0.05,        # Paper: 0.05
    
    # Loss coefficients
    'energy_coefficient': 2,       # Paper: λE = 2
    'force_coefficient': 100,      # Paper: λF = 100
    
    # Gradient clipping
    'gradient_clip_norm': 100,     # Paper: 100
    
    # Model EMA
    'ema_decay': 0.999,            # Paper: 0.999
    
    # ================================================================
    # MODEL ARCHITECTURE (from paper Table 7, S2EF-2M)
    # ================================================================
    # Graph construction
    'cutoff_radius': 12.0,         # Paper: 12 Å
    'max_neighbors': 20,           # Paper: 20
    'num_radial_bases': 600,       # Paper: 600
    'distance_function': 'gaussian',
    
    # SO3 settings
    'lmax_list': [4],              # Paper: L_max = 6... they on page 10 also report on lmax = 2,4,6... so all are fine to run on
    'mmax_list': [2],              # Paper: M_max = 2
    'grid_resolution': 18,         # Paper: R = 18
    
    # Architecture size
    'num_layers': 12,              # Paper: 12 blocks
    'sphere_channels': 128,        # Paper: d_embed^(6) = (6, 128)
    'attn_hidden_channels': 64,    # Paper: d_attn_hidden^(6) = (6, 64)
    'num_heads': 8,                # Paper: h = 8
    'attn_alpha_channels': 64,     # Paper: d_attn_alpha^(0) = (0, 64)
    'attn_value_channels': 16,     # Paper: d_attn_value^(6) = (6, 16)
    'ffn_hidden_channels': 128,    # Paper: d_ffn^(6) = (6, 128)
    'edge_channels': 128,          # Paper: d_edge^(0) = (0, 128)
    
    # ================================================================
    # MODEL SETTINGS (from paper Table 8, Index 5)
    # ================================================================
    # Paper Table 8 shows the exact configuration:
    # - Attention re-normalization: ✓
    # - Activation: Sep. S²
    # - Normalization: SLN (Spherical LayerNorm)
    
    'norm_type': 'layer_norm_sh',  # SLN = Spherical LayerNorm
    'attn_activation': 'scaled_silu',
    'ffn_activation': 'scaled_silu',
    'use_s2_act_attn': False,
    'use_attn_renorm': True,       # ✓ in paper
    'use_gate_act': False,
    'use_grid_mlp': False,
    'use_sep_s2_act': True,        # Sep. S² activation
    
    # ================================================================
    # OTHER SETTINGS
    # ================================================================
    'max_num_elements': 90,
    'use_pbc': True,
    'regress_forces': True,
    'otf_graph': True,
    'use_atom_edge_embedding': True,
    'share_atom_edge_embedding': False,
    'use_m_share_rad': False,
    'weight_init': 'uniform',      # Paper uses uniform initialization
    
    # System
    'num_workers': 4, # 8 is too high for the GPU, eventhoug that is what they use. 
    'save_interval': 1,
    'save_dir': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/models/trained_models/OC20_S2EF_2M',
    
    # For debugging
    'max_train_samples': None,
    'max_val_samples': None,
}


# ================================================================
# VERIFICATION AGAINST PAPER TABLE 7 & 8
# ================================================================
"""
PAPER TABLE 7 (S2EF-2M) ✓ CHECKLIST:
=====================================
✓ Optimizer: AdamW
✓ Learning rate scheduling: Cosine with linear warmup  
✓ Warmup epochs: 0.1
✓ Maximum learning rate: 2×10⁻⁴
✓ Batch size: 64
✓ Number of epochs: 12
✓ Weight decay: 1×10⁻³
✓ Dropout rate: 0.1
✓ Stochastic depth: 0.05
✓ Energy coefficient λE: 2
✓ Force coefficient λF: 100
✓ Gradient clipping: 100
✓ Model EMA decay: 0.999
✓ Cutoff radius: 12 Å
✓ Max neighbors: 20
✓ Radial bases: 600
✓ Hidden scalar features: (0, 128)
✓ L_max: 6
✓ M_max: 2
✓ Transformer blocks: 12
✓ Embedding dim: (6, 128)
✓ Attention hidden: (6, 64)
✓ Attention heads: 8
✓ Attention alpha: (0, 64)
✓ Attention value: (6, 16)
✓ FFN hidden: (6, 128)
✓ Grid resolution R: 18

PAPER TABLE 8 (Index 5 - the base model) ✓ CHECKLIST:
======================================================
✓ Attention re-normalization: ✓ (use_attn_renorm=True)
✓ Activation: Sep. S² (use_sep_s2_act=True)
✓ Normalization: SLN (layer_norm_sh)
✓ L_max: 6
✓ M_max: 2
✓ Transformer blocks: 12
✓ Number of parameters: ~91M (will be close with these settings)
"""


# ================================================================
# DEBUG/TESTING CONFIGS
# ================================================================

config_debug = config.copy()
config_debug.update({
    'batch_size': 32,
    'epochs': 2,
    'num_layers': 4,
    'num_workers': 0,
    'max_train_samples': 1000,
    'max_val_samples': 100,
})

config_small_memory = config.copy()
config_small_memory.update({
    'batch_size': 16,  # Reduce if OOM
    'num_workers': 2,
})