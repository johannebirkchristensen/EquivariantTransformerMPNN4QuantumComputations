"""
MatPES-PBE Configuration for EquiformerV2
==========================================
Task: Universal Machine Learning Interatomic Potential (MLIP)
      Predicts energy, forces, and stress for periodic crystals.

Dataset: MatPES-PBE v2025.1 (~400k structures from MD sampling)
         Source: https://github.com/materialyzeai/matpes
         HuggingFace: mavrl/matpes  (config="pbe")

HOW TO DOWNLOAD
---------------
Option A — matpes CLI (recommended):
    pip install matpes
    matpes download pbe          # → MatPES-PBE-20240214.json.gz  (~2 GB)

Option B — HuggingFace datasets:
    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("mavrl/matpes", "pbe")

Option C — MPContribs (bulk download, requires MP API key):
    See https://docs.materialsproject.org/collaborations/matpes

LEARNING TARGETS (per structure entry)
---------------------------------------
  energy_per_atom   eV/atom    Total DFT energy / N_atoms (PBE functional)
  forces            eV/Å       Per-atom force vectors [N, 3]
  stress            kBar       3×3 Voigt stress tensor on the unit cell
                               (converted to eV/Å³ in training = kBar / 1602.1766)

  (magmom — per-atom magnetic moments in μB — present but optional)

EVALUATION METRICS (Table 1 in arXiv:2503.04070)
--------------------------------------------------
  Energy MAE:  ~3–5 meV/atom
  Force  MAE:  ~50–80 meV/Å
  Stress MAE:  ~0.3–0.6 GPa

These are typical numbers for TensorNet/M3GNet trained on MatPES.
EquiformerV2 should be competitive or better given its equivariant design.

LOSS FUNCTION
-------------
  L = w_e * MAE(energy) + w_f * MAE(forces) + w_s * MAE(stress)
  Typical weights: w_e=1.0, w_f=1.0, w_s=0.1
  (forces dominate by count: N forces vs 6 stress components vs 1 energy)
"""

import os

_BASE = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations'

config = {

    # ── Paths ──────────────────────────────────────────────────────
    'data_path':   '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/MatPES/MatPES-PBE-2025.1.json.gz',
    'cache_dir':   os.path.join(_BASE, 'datasets/MatPES/cache'),
    'save_dir':    os.path.join(_BASE, 'models/trained_models/MatPES/baselineEquiformerV2'),

    # ── Task ──────────────────────────────────────────────────────
    'use_pbc':          True,
    'regress_forces':   True,    # ← KEY: forces are the primary signal
    'regress_stress':   False,   # ← stress tensor (6 components Voigt)
    'regress_magmom':   False,   # optional; skip for first run

    # ── Loss weights ──────────────────────────────────────────────
    # Energy in eV/atom, forces in eV/Å, stress in eV/Å³
    # Forces are numerically ~10-100× larger per-sample → weight them lower
    'energy_loss_weight': 1.0,
    'force_loss_weight':  1.0,
    'stress_loss_weight': 0.1,

    # ── Training ──────────────────────────────────────────────────
    'epochs':               30,        # PES datasets converge faster
    'batch_size':           8,         # smaller: each structure has ~30 atoms → big graphs... 8 is max on l40s
    'learning_rate':        0.0002,
    'weight_decay':         0.01,
    'gradient_clip_norm':   5.0,
    'loss_function':        'L1',      # MAE preferred for force fields
    'use_mixed_precision':  False,

    # ── Learning rate scheduler ────────────────────────────────────
    # CosineAnnealingLR decays lr from learning_rate → lr_min over all epochs.
    # No warmup: the scheduler starts at full learning_rate from epoch 1.
    'lr_scheduler':  'cosine',        # informational label
    'lr_min':        0.0,             # eta_min for CosineAnnealingLR

    # ── Train/val/test split ──────────────────────────────────────
    # MatPES has no official split; we use 90/5/5
    'train_frac': 0.90,
    'val_frac':   0.05,
    'test_frac':  0.05,
    'random_seed': 42,

    # ── Graph construction ─────────────────────────────────────────
    'cutoff_radius': 6.0,        # Å — same as MP20
    'max_neighbors': 20,

    # ── Architecture ──────────────────────────────────────────────
    # Same backbone as MP20 but regress_forces=True adds gradient computation
    'num_layers':             6,
    'lmax_list':              [4],
    'mmax_list':              [2],
    'sphere_channels':        128,
    'attn_hidden_channels':   128,
    'num_heads':              8,
    'attn_alpha_channels':    32,
    'attn_value_channels':    16,
    'ffn_hidden_channels':    512,
    'edge_channels':          128,
    'grid_resolution':        18,
    'max_num_elements':       100,     # full periodic table

    # ── Model settings ─────────────────────────────────────────────
    'norm_type':                  'rms_norm_sh',
    'distance_function':          'gaussian',
    'num_radial_bases':           512,
    'use_atom_edge_embedding':    True,
    'share_atom_edge_embedding':  False,
    'use_m_share_rad':            False,
    'attn_activation':            'scaled_silu',
    'use_s2_act_attn':            False,
    'use_attn_renorm':            True,
    'ffn_activation':             'scaled_silu',
    'use_gate_act':               False,
    'use_grid_mlp':               False,
    'use_sep_s2_act':             True,
    'alpha_drop':                 0.05,    # less dropout: force labels are noisy
    'drop_path_rate':             0.05,
    'proj_drop':                  0.0,
    'weight_init':                'normal',

    # ── Data loading ───────────────────────────────────────────────
    'num_workers':      4,
    'max_train':        None,      # None = use full split
    'max_val':          None,
    'max_test':         None,

    # ── Checkpointing ──────────────────────────────────────────────
    'save_interval':    10,
    'resume_from':      None,

    # ── Logging ────────────────────────────────────────────────────
    'log_interval':     10,

    # ── Functional ─────────────────────────────────────────────────
    'functional': 'pbe',   # 'pbe' or 'r2scan'
}

# ════════════════════════════════════════════════════════════════
# UNIT NOTES
# ════════════════════════════════════════════════════════════════
"""
Raw JSON fields:
  energy          [eV]      total DFT energy
  energy_per_atom [eV/atom] = energy / N_atoms     ← used for training
  force           [eV/Å]    list of [fx, fy, fz]  ← used for training
  stress          [kBar]    3×3 or Voigt-6          ← converted to eV/Å³

Conversion:
  1 kBar = 0.1 GPa = 0.1 / 160.2176621 eV/Å³
  kBar_to_eV_Ang3 = 1.0 / 1602.1766

Stress Voigt convention: [σxx, σyy, σzz, σyz, σxz, σxy]
"""

KBAR_TO_EV_ANG3 = 1.0 / 1602.1766   # multiply stress (kBar) by this


