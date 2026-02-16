"""
QM9 Data Loader - FINAL CORRECTED VERSION
==========================================
For PyTorch Geometric QM9 dataset stored in ASE database

PyTorch Geometric QM9 units (from documentation):
- μ: Debye (D)
- α: Bohr³ (a₀³)
- Energies (ε_HOMO, ε_LUMO, Δε, ZPVE): eV
- Energies (U₀, U, H, G): kcal/mol
- R²: Bohr² (a₀²)
- C_v: cal/(mol·K)

Paper reports in:
- Energies: meV
- Everything else: same units
"""
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.QM9.config_equiformerV2_mu_alpha_homo_lumo_osv import config
from ase.db import connect
import torch
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
import numpy as np

DB_PATH = config.get('db_path', 'qm9.db')

# ============================================================================
# UNIT CONVERSION CONSTANTS
# ============================================================================
EV_TO_MEV = 1000            # eV to meV.. already in meV from dataload. 

# ============================================================================
# QM9 NORMALIZATION STATISTICS
# ============================================================================
QM9_NORMALIZE = True
print("QM9_NORMALIZE:", QM9_NORMALIZE)

import json

# expected paper-order property names
PROPERTY_ORDER = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                  'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']

def load_qm9_stats_from_json(json_path):
    """
    Load QM9 stats JSON and return QM9_TARGET_MEAN, QM9_TARGET_STD
    as torch.float32 tensors in PAPER ORDER.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Stats JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        j = json.load(f)

    stats = j.get('stats', j)
    means = []
    stds  = []
    missing = []
    
    for name in PROPERTY_ORDER:
        if name in stats:
            entry = stats[name]
        else:
            # try to match by normalized ascii fallback
            found = None
            for k in stats.keys():
                if k == name:
                    found = k
                    break
                if str(k).replace(" ", "") == str(name).replace(" ", ""):
                    found = k
                    break
            if found is None:
                missing.append(name)
                continue
            entry = stats[found]

        means.append(float(entry['mean']))
        stds.append(float(entry['std']))

    if missing:
        raise RuntimeError(f"Missing properties in stats JSON for: {missing}")

    QM9_TARGET_MEAN = torch.tensor(means, dtype=torch.float32)
    QM9_TARGET_STD  = torch.tensor(stds, dtype=torch.float32)

    # safety: avoid zeros in std
    QM9_TARGET_STD[QM9_TARGET_STD == 0.0] = 1.0

    # quick sanity print
    print("Loaded QM9 stats from:", json_path)
    for n, m, s in zip(PROPERTY_ORDER, QM9_TARGET_MEAN.tolist(), QM9_TARGET_STD.tolist()):
        print(f"  {n:6s}: mean = {m:12.6f}, std = {s:12.6f}")

    return QM9_TARGET_MEAN, QM9_TARGET_STD

# Load stats from JSON file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATS_JSON = os.path.join(
    BASE_DIR,
    'datasets', 'QM9', 'DatasetStastics', 'run_stats_corrected', 'qm9_full_stats.json'
)

QM9_TARGET_MEAN, QM9_TARGET_STD = load_qm9_stats_from_json(STATS_JSON)


class QM9Dataset(Dataset):
    """
    QM9 Dataset from PyTorch Geometric (via ASE database)
    
    Properties in PAPER ORDER:
    0. α - Isotropic polarizability (Bohr³)
    1. Δε - HOMO-LUMO gap (meV)
    2. ε_HOMO - HOMO energy (meV)
    3. ε_LUMO - LUMO energy (meV)
    4. μ - Dipole moment (Debye)
    5. C_v - Heat capacity (cal/mol/K)
    6. G - Free energy at 298.15K (meV)
    7. H - Enthalpy at 298.15K (meV)
    8. R² - Electronic spatial extent (Bohr²)
    9. U - Internal energy at 298.15K (meV)
    10. U₀ - Internal energy at 0K (meV)
    11. ZPVE - Zero point vibrational energy (meV)
    """
    
    def __init__(self, db_path, transform=None, max_samples=None, normalize=True, target_indices=None):
        """
        Args:
            db_path: Path to ASE database created from PyG QM9
            transform: Optional transform
            max_samples: Limit dataset size
            normalize: Whether to normalize targets
            target_indices: List of target indices to select (e.g., [6,7,9,10] for G,H,U,U₀)
                           If None, returns all 12 properties
        """
        self.db = connect(db_path)
        self.normalize = normalize
        self.target_indices = target_indices
        
        n = self.db.count()
        valid_ids = list(range(1, n + 1))
        
        if max_samples is not None:
            valid_ids = valid_ids[:max_samples]
        
        self.keys = valid_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        """
        Returns sample with targets in PAPER ORDER with PAPER UNITS
        """
        key = self.keys[idx]
        
        try:
            row = self.db.get(id=key)
        except KeyError:
            print(f"ERROR: Invalid key: {key}, idx: {idx}")
            raise
        
        atoms = row.toatoms()
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # ================================================================
        # EXTRACT VALUES (PyTorch Geometric QM9 units from database)
        # ================================================================
        try:
            mu_raw = row.data['mu']        # Debye
            alpha_raw = row.data['alpha']  # Bohr³
            homo_raw = row.data['homo']    # eV
            lumo_raw = row.data['lumo']    # eV
            gap_raw = row.data['gap']      # eV
            r2_raw = row.data['r2']        # Bohr²
            zpve_raw = row.data['zpve']    # eV
            u0_raw = row.data['U0']        # kcal/mol
            u_raw = row.data['U']          # kcal/mol
            h_raw = row.data['H']          # kcal/mol
            g_raw = row.data['G']          # kcal/mol
            cv_raw = row.data['Cv']        # cal/mol/K
            
        except KeyError as e:
            print(f"ERROR: Missing property {e} in row {key}")
            print(f"Available keys: {list(row.data.keys())}")
            raise
        
        # ================================================================
        # CONVERT TO PAPER UNITS AND REORDER TO PAPER ORDER
        # ================================================================
        # Paper order: α, Δε, ε_HOMO, ε_LUMO, μ, C_v, G, H, R², U, U₀, ZPVE
        targets = torch.tensor([
            alpha_raw,                       # 0. α (Bohr³)
            gap_raw * EV_TO_MEV,            # 1. Δε (eV → meV)
            homo_raw * EV_TO_MEV,           # 2. ε_HOMO (eV → meV)
            lumo_raw * EV_TO_MEV,           # 3. ε_LUMO (eV → meV)
            mu_raw,                         # 4. μ (Debye)
            cv_raw,                         # 5. C_v (cal/mol/K)
            g_raw * EV_TO_MEV,              # 6. G eV → meV)
            h_raw * EV_TO_MEV,              # 7. H (eV → meV)
            r2_raw,                         # 8. R² (Bohr²)
            u_raw * EV_TO_MEV,              # 9. U (eV → meV)
            u0_raw * EV_TO_MEV,             # 10. U₀ (eV → meV)
            zpve_raw * EV_TO_MEV,           # 11. ZPVE (eV → meV)
        ], dtype=torch.float32)
        
        # ================================================================
        # SELECT SPECIFIC TARGETS (if requested)
        # ================================================================
        if self.target_indices is not None:
            targets = targets[self.target_indices]
            # Normalize with corresponding mean/std
            if self.normalize and QM9_NORMALIZE:
                mean = QM9_TARGET_MEAN[self.target_indices]
                std = QM9_TARGET_STD[self.target_indices]
                targets = (targets - mean) / std
        elif self.normalize and QM9_NORMALIZE:
            # Normalize all 12 targets
            targets = (targets - QM9_TARGET_MEAN) / QM9_TARGET_STD
        
        # Create sample
        sample = {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'natoms': len(atomic_numbers),
            'targets': targets,
            'batch': torch.zeros(len(atomic_numbers), dtype=torch.long)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_mol(batch):
    """Collate function for batching"""
    atomic_numbers = torch.cat([b['atomic_numbers'] for b in batch], dim=0)
    pos = torch.cat([b['pos'] for b in batch], dim=0)
    natoms = torch.tensor([b['natoms'] for b in batch], dtype=torch.long)
    batch_tensor = torch.cat([b['batch'] + i for i, b in enumerate(batch)], dim=0)
    targets = torch.stack([b['targets'] for b in batch], dim=0)
    
    return {
        'atomic_numbers': atomic_numbers,
        'pos': pos,
        'natoms': natoms,
        'batch': batch_tensor,
        'targets': targets
    }


def get_qm9_loaders(
    db_path,
    batch_size=16,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
    max_samples=None,
    num_workers=0,
    normalize=True,
    target_indices=None
):
    """Create train/val/test loaders"""
    dataset = QM9Dataset(
        db_path, 
        max_samples=max_samples, 
        normalize=normalize,
        target_indices=target_indices
    )
    
    total = len(dataset)
    n_val = int(val_split * total)
    n_test = int(test_split * total)
    n_train = total - n_val - n_test
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_mol, num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mol, num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=collate_mol, num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader


    """Convert normalized targets back to paper units"""
    if not QM9_NORMALIZE:
        return targets_normalized
    
    if target_indices is not None:
        mean = QM9_TARGET_MEAN[target_indices]
        std = QM9_TARGET_STD[target_indices]
    else:
        mean = QM9_TARGET_MEAN
        std = QM9_TARGET_STD
    
    return targets_normalized * std + mean

def denormalize_targets(targets_normalized, target_indices=None):
    """Convert normalized targets back to paper units"""
    if not QM9_NORMALIZE:
        return targets_normalized
    
    if target_indices is not None:
        # Select the corresponding mean and std for the target indices
        mean = QM9_TARGET_MEAN[target_indices]
        std = QM9_TARGET_STD[target_indices]
    else:
        mean = QM9_TARGET_MEAN
        std = QM9_TARGET_STD
    
    # Make sure mean and std are on the same device as targets
    mean = mean.to(targets_normalized.device)
    std = std.to(targets_normalized.device)
    
    return targets_normalized * std + mean