"""
QM9 Data Loader - FINAL CORRECTED VERSION
==========================================
For PyTorch Geometric QM9 dataset stored in ASE database

PyTorch Geometric QM9 units (from documentation):
- μ: Debye (D)
- α: Bohr³ (a₀³)
- Energies (ε_HOMO, ε_LUMO, Δε, ZPVE, U₀, U, H, G): eV
- R²: Bohr² (a₀²)
- C_v: cal/(mol·K)

Paper reports in:
- Energies: meV (so we need ×1000 for eV → meV)
- Everything else: same units
"""
import sys 
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.QM9.config_equiformerV2 import config
from ase.db import connect
import torch
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
import numpy as np
DB_PATH = config.get('db_path', 'qm9.db')

# ============================================================================
# UNIT CONVERSION CONSTANTS
# ============================================================================
EV_TO_MEV = 1000.0  # PyG QM9 stores energies in eV, paper uses meV


# ============================================================================
# QM9 NORMALIZATION STATISTICS
# ============================================================================
# Mean and std for each property (in paper units: meV, Debye, Bohr³, etc.)
# Order: α, Δε, ε_HOMO, ε_LUMO, μ, C_v, G, H, R², U, U₀, ZPVE

QM9_NORMALIZE = True  # Set to False to disable normalization
print("QM9_NORMALIZE:", QM9_NORMALIZE)
import json
import numpy as np
import torch
import os

# expected paper-order property names (use the same labels your stats JSON used)
PROPERTY_ORDER = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                  'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']

def load_qm9_stats_from_json(json_path):
    """
    Load QM9 stats JSON (as produced by compute_qm9_stats) and return
    QM9_TARGET_MEAN, QM9_TARGET_STD as torch.float32 tensors in PAPER ORDER.

    Args:
        json_path (str): path to the stats JSON file.

    Returns:
        QM9_TARGET_MEAN (torch.Tensor, shape [12], dtype float32)
        QM9_TARGET_STD  (torch.Tensor, shape [12], dtype float32)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Stats JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        j = json.load(f)

    stats = j.get('stats', j)  # allow passing a dict directly
    # build arrays in desired order
    means = []
    stds  = []
    missing = []
    for name in PROPERTY_ORDER:
        # JSON may have unicode-escaped names; ensure exact match
        if name in stats:
            entry = stats[name]
        else:
            # try to match by normalized ascii fallback (e.g. map 'α' <-> '\\u03b1')
            found = None
            for k in stats.keys():
                if k == name:
                    found = k; break
                # fallback: compare normalized NFKD strings (strip combining marks)
                if str(k).replace(" ", "") == str(name).replace(" ", ""):
                    found = k; break
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
# Load stats from JSON file (adjust path as needed)
# base directory of your project (two levels up from data_loader_qm9_v3.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATS_JSON = os.path.join(
    BASE_DIR,
    'datasets', 'QM9', 'DatasetStastics', 'run_stats', 'qm9_full_stats.json'
)

QM9_TARGET_MEAN, QM9_TARGET_STD = load_qm9_stats_from_json(STATS_JSON)

"""
QM9_TARGET_MEAN = torch.tensor([
    75.0,      # α (Bohr³)
    5000.0,    # Δε (meV)
    -7000.0,   # ε_HOMO (meV)
    -1000.0,   # ε_LUMO (meV)
    2.0,       # μ (Debye)
    7.0,       # C_v (cal/mol/K)
    -60000.0,  # G (meV)
    -58000.0,  # H (meV)
    1200.0,    # R² (Bohr²)
    -58000.0,  # U (meV)
    -58000.0,  # U₀ (meV)
    100.0,     # ZPVE (meV)
], dtype=torch.float32)

QM9_TARGET_STD = torch.tensor([
    10.0,    # α
    2000.0,  # Δε
    1000.0,  # ε_HOMO
    1000.0,  # ε_LUMO
    1.5,     # μ
    1.5,     # C_v
    5000.0,  # G
    5000.0,  # H
    200.0,   # R²
    5000.0,  # U
    5000.0,  # U₀
    20.0,    # ZPVE
], dtype=torch.float32)
"""

class QM9Dataset(Dataset):
    """
    QM9 Dataset from PyTorch Geometric (via ASE database)
    
    Properties in PAPER ORDER:
    1. α - Isotropic polarizability (Bohr³)
    2. Δε - HOMO-LUMO gap (meV)
    3. ε_HOMO - HOMO energy (meV)
    4. ε_LUMO - LUMO energy (meV)
    5. μ - Dipole moment (Debye)
    6. C_v - Heat capacity (cal/mol/K)
    7. G - Free energy at 298.15K (meV)
    8. H - Enthalpy at 298.15K (meV)
    9. R² - Electronic spatial extent (Bohr²)
    10. U - Internal energy at 298.15K (meV)
    11. U₀ - Internal energy at 0K (meV)
    12. ZPVE - Zero point vibrational energy (meV)
    """
    
    def __init__(self, db_path, transform=None, max_samples=None, normalize=True):
        """
        Args:
            db_path: Path to ASE database created from PyG QM9
            transform: Optional transform
            max_samples: Limit dataset size
            normalize: Whether to normalize targets
        """
        self.db = connect(db_path)
        self.normalize = normalize
        
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
        # EXTRACT VALUES (PyTorch Geometric QM9 units from your database)
        # ================================================================
        try:
            # From PyG QM9 - targets are ordered as:
            # 0=mu, 1=alpha, 2=homo, 3=lumo, 4=gap, 5=r2, 6=zpve,
            # 7=U0, 8=U, 9=H, 10=G, 11=Cv
            
            mu_raw = row.data['mu']        # Debye (already correct!)
            alpha_raw = row.data['alpha']  # Bohr³ (already correct!)
            homo_raw = row.data['homo']    # eV (need → meV)
            lumo_raw = row.data['lumo']    # eV (need → meV)
            gap_raw = row.data['gap']      # eV (need → meV)
            r2_raw = row.data['r2']        # Bohr² (already correct!)
            zpve_raw = row.data['zpve']    # eV (need → meV)
            u0_raw = row.data['U0']        # eV (need → meV)
            u_raw = row.data['U']          # eV (need → meV)
            h_raw = row.data['H']          # eV (need → meV)
            g_raw = row.data['G']          # eV (need → meV)
            cv_raw = row.data['Cv']        # cal/mol/K (already correct!)
            
        except KeyError as e:
            print(f"ERROR: Missing property {e} in row {key}")
            print(f"Available keys: {list(row.data.keys())}")
            raise
        
        # ================================================================
        # CONVERT TO PAPER UNITS AND REORDER TO PAPER ORDER
        # ================================================================
        # Paper order: α, Δε, ε_HOMO, ε_LUMO, μ, C_v, G, H, R², U, U₀, ZPVE
        
        targets = torch.tensor([
            alpha_raw,                  # 1. α (Bohr³) - NO conversion needed
            gap_raw * EV_TO_MEV,        # 2. Δε (eV → meV)
            homo_raw * EV_TO_MEV,       # 3. ε_HOMO (eV → meV)
            lumo_raw * EV_TO_MEV,       # 4. ε_LUMO (eV → meV)
            mu_raw,                     # 5. μ (Debye) - NO conversion needed
            cv_raw,                     # 6. C_v (cal/mol/K) - NO conversion needed
            g_raw * EV_TO_MEV,          # 7. G (eV → meV)
            h_raw * EV_TO_MEV,          # 8. H (eV → meV)
            r2_raw,                     # 9. R² (Bohr²) - NO conversion needed
            u_raw * EV_TO_MEV,          # 10. U (eV → meV)
            u0_raw * EV_TO_MEV,         # 11. U₀ (eV → meV)
            zpve_raw * EV_TO_MEV,       # 12. ZPVE (eV → meV)
        ], dtype=torch.float32)
        
        # ================================================================
        # NORMALIZE (if enabled)
        # ================================================================
        if self.normalize and QM9_NORMALIZE:
            targets = (targets - QM9_TARGET_MEAN) / QM9_TARGET_STD
        
        # Create sample
        sample = {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'natoms': len(atomic_numbers),
            'targets': targets,  # [12] in CORRECT order, units, normalized
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
        'targets': targets  # [batch_size, 12]
    }


def get_qm9_loaders(
    db_path,
    batch_size=16,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
    max_samples=None,
    num_workers=0,
    normalize=True
):
    """Create train/val/test loaders"""
    dataset = QM9Dataset(db_path, max_samples=max_samples, normalize=normalize)
    
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


def denormalize_targets(targets_normalized):
    """Convert normalized targets back to paper units"""
    if not QM9_NORMALIZE:
        return targets_normalized
    return targets_normalized * QM9_TARGET_STD + QM9_TARGET_MEAN


# ================================================================
# TESTING
# ================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader_qm9_final.py <path_to_qm9.db>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    print("="*70)
    print("Testing FINAL CORRECTED QM9 Data Loader")
    print("(For PyTorch Geometric QM9 dataset)")
    print("="*70)
    
    train_loader, val_loader, test_loader = get_qm9_loaders(
        db_path,
        batch_size=4,
        max_samples=100,
        num_workers=0,
        normalize=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val:   {len(val_loader.dataset)}")
    print(f"  Test:  {len(test_loader.dataset)}")
    
    batch = next(iter(train_loader))
    
    print(f"\nBatch shape:")
    print(f"  targets: {batch['targets'].shape}  ← Should be [4, 12]")
    
    if batch['targets'].shape == (4, 12):
        print("✓ Shape correct!")
    
    # Show normalized targets
    print(f"\nFirst molecule targets (NORMALIZED):")
    property_names = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                      'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']
    for i, (name, value) in enumerate(zip(property_names, batch['targets'][0]), 1):
        print(f"  {i:2d}. {name:8s}: {value:10.4f}  (normalized)")
    
    # Denormalize and show
    targets_denorm = denormalize_targets(batch['targets'])
    print(f"\nFirst molecule targets (DENORMALIZED, paper units):")
    units = ['Bohr³', 'meV', 'meV', 'meV', 'Debye', 'cal/mol/K',
             'meV', 'meV', 'Bohr²', 'meV', 'meV', 'meV']
    for i, (name, value, unit) in enumerate(zip(property_names, targets_denorm[0], units), 1):
        print(f"  {i:2d}. {name:8s}: {value:10.2f} {unit}")
    
    # Check ranges
    print(f"\n" + "="*70)
    print("UNIT & ORDER CHECK")
    print("="*70)
    
    t = targets_denorm[0]
    checks = [
        ("α", t[0], 50, 200, "Bohr³"),
        ("Δε", t[1], 1000, 10000, "meV"),
        ("ε_HOMO", t[2], -10000, -2000, "meV"),
        ("ε_LUMO", t[3], -2000, 2000, "meV"),
        ("μ", t[4], 0, 6, "Debye"),
        ("C_v", t[5], 5, 15, "cal/mol/K"),
        ("ZPVE", t[11], 50, 150, "meV"),
    ]
    
    all_ok = True
    for name, value, min_val, max_val, unit in checks:
        if min_val <= value <= max_val:
            print(f"✓ {name:8s} = {value:10.2f} {unit:12s} [OK]")
        else:
            print(f"⚠ {name:8s} = {value:10.2f} {unit:12s} [OUT OF RANGE]")
            all_ok = False
    
    if all_ok:
        print(f"\n✓✓✓ All checks passed! Data loader is correct!")
    else:
        print(f"\n⚠ Some values out of range - check database")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)