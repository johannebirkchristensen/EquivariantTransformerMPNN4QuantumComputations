# Recommended starting config for QM9
config = {
    'use_pbc': False,  # No periodic boundaries
    'regress_forces': False,  # Not needed for QM9
    'otf_graph': True,  # Compute graph on-the-fly
    'max_neighbors': 50,  # ~18 atoms, so lower than OC20
    'max_radius': 5.0,  # Ångströms
    'max_num_elements': 10,  # QM9 only has H, C, N, O, F
    
    'num_layers': 8,  # Can start smaller (4-6) for experiments
    'sphere_channels': 128,
    'attn_hidden_channels': 128,
    'num_heads': 8,
    'attn_alpha_channels': 32,
    'attn_value_channels': 16,
    'ffn_hidden_channels': 512,
    
    'lmax_list': [4],  # Lower than OC20's [6], sufficient for small molecules
    'mmax_list': [2],  # Order truncation
    
    'num_targets': 12,  # QM9 has 12 properties

    'batch_size': 16,
    'learning_rate': 1e-3,
    'epochs': 50,
    'db_path': "/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db"

}

