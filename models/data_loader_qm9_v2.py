"""
QM9 Data Loader - Fixed to return ALL 12 targets
==================================================
FIXED: Now returns all 12 QM9 properties, not just 1
"""

from time import time
from ase.db import connect
import os
import torch
from torch.utils.data import Dataset, DataLoader
from ase import Atoms


class QM9Dataset(Dataset):
    """
    QM9 Dataset returning all 12 molecular properties
    
    Properties in PAPER ORDER (EquiformerV2 Table):
    1. α - Polarizability
    2. Δε - HOMO-LUMO gap
    3. ε_HOMO - HOMO energy
    4. ε_LUMO - LUMO energy
    5. μ - Dipole moment
    6. C_v - Heat capacity at 298K
    7. G - Free energy at 298K
    8. H - Enthalpy at 298K
    9. R² - Electronic spatial extent
    10. U - Internal energy at 298K
    11. U₀ - Internal energy at 0K
    12. ZPVE - Zero point vibrational energy
    """
    
    def __init__(self, db_path, transform=None, max_samples=None):
        """
        Args:
            db_path (str): Path to QM9 ASE database
            transform (callable, optional): Optional transform
            max_samples (int, optional): Limit dataset size for debugging
        """
        self.db = connect(db_path)
        
        # Get all valid row IDs (ASE DB is 1-indexed)
        n = self.db.count()
        valid_ids = list(range(1, n + 1))
        
        # Limit for debugging if requested
        if max_samples is not None:
            valid_ids = valid_ids[:max_samples]
        
        self.keys = valid_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.keys)
    # --- robust key lookup helper ---
    def _get_row_value(row, candidates):
        """
        Try multiple candidate keys in row.data and return the first found.
        Raises KeyError listing available keys if none found.
        """
        for k in candidates:
            if k in row.data:
                return row.data[k]
        raise KeyError(f"None of {candidates} found in row.data. Available keys: {list(row.data.keys())}")

    
    def __getitem__(self, idx):
        """
        Returns a sample dictionary with all 12 QM9 targets
        """
        key = self.keys[idx]
        
        try:
            row = self.db.get(id=key)
        except KeyError:
            print(f"ERROR: Invalid key: {key}, idx: {idx}")
            raise
        
        atoms = row.toatoms()
        
        # Atomic numbers and positions
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        
        # ============================================================
        # CRITICAL: Return ALL 12 targets in PAPER ORDER
        # ============================================================
        # Order from EquiformerV2 paper Table:
        # α, Δε, ε_HOMO, ε_LUMO, μ, C_v, G, H, R², U, U₀, ZPVE
    
        try:
            targets_list = torch.tensor([
                row.data['mu'],      # 0. μ  - Dipole moment
                row.data['alpha'],   # 1. α  - Isotropic polarizability
                row.data['homo'],    # 2. ε_HOMO
                row.data['lumo'],    # 3. ε_LUMO
                row.data['gap'],     # 4. Δε - HOMO–LUMO gap
                row.data['r2'],      # 5. R² - Electronic spatial extent
                row.data['zpve'],    # 6. ZPVE
                row.data['U0'],      # 7. U₀ - Internal energy at 0K
                row.data['U'],       # 8. U  - Internal energy at 298.15K
                row.data['H'],       # 9. H  - Enthalpy at 298.15K
                row.data['G'],       # 10. G - Free energy at 298.15K
                row.data['Cv'],      # 11. C_v - Heat capacity at 298.15K
            ], dtype=torch.float32)
        except KeyError as e:
            print(f"ERROR: Missing property {e} in database row {key}")
            print(f"Available keys: {row.data.keys()}")
            raise
         
        targets = torch.tensor(targets_list, dtype=torch.float32)

        # === convert energy quantities from eV -> meV to match EquiformerV2 ===
        EV_TO_MEV = 1000.0
        # zero-based indices of targets that are in eV and should become meV:
        ev_indices = [1, 2, 3, 6, 7, 9, 10, 11]
        targets[ev_indices] = targets[ev_indices] * EV_TO_MEV

        # Create sample dictionary
        sample = {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'natoms': len(atomic_numbers),
            'targets': targets,  # Shape: [12]
            'batch': torch.zeros(len(atomic_numbers), dtype=torch.long)
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


def collate_mol(batch):
    """
    Collate function to batch multiple molecules together
    
    Returns dictionary with batched tensors
    """
    atomic_numbers = torch.cat([b['atomic_numbers'] for b in batch], dim=0)
    pos = torch.cat([b['pos'] for b in batch], dim=0)
    natoms = torch.tensor([b['natoms'] for b in batch], dtype=torch.long)
    batch_tensor = torch.cat([b['batch'] + i for i, b in enumerate(batch)], dim=0)
    
    # Stack targets - shape: [batch_size, 12]
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
    num_workers=0
):
    """
    Create train/val/test data loaders for QM9
    
    Args:
        db_path (str): Path to QM9 database
        batch_size (int): Batch size
        val_split (float): Fraction for validation
        test_split (float): Fraction for test
        shuffle (bool): Shuffle training data
        max_samples (int): Limit dataset size (None = full dataset)
        num_workers (int): Number of workers (0 for debugging)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset
    dataset = QM9Dataset(db_path, max_samples=max_samples)
    
    total = len(dataset)
    n_val = int(val_split * total)
    n_test = int(test_split * total)
    n_train = total - n_val - n_test
    
    # Split dataset
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)  # Only pin if using workers
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mol,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader


# ================================================================
# TESTING
# ================================================================
if __name__ == "__main__":
    """
    Test the data loader to make sure it works correctly
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader_qm9_fixed.py <path_to_qm9.db>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    print("="*60)
    print("Testing QM9 Data Loader")
    print("="*60)
    
    # Create loaders with small sample for testing
    train_loader, val_loader, test_loader = get_qm9_loaders(
        db_path,
        batch_size=4,
        max_samples=100,
        num_workers=0
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)}")
    print(f"  Val:   {len(val_loader.dataset)}")
    print(f"  Test:  {len(test_loader.dataset)}")
    
    # Test one batch
    print(f"\nTesting one batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  atomic_numbers: {batch['atomic_numbers'].shape}")
    print(f"  pos:            {batch['pos'].shape}")
    print(f"  natoms:         {batch['natoms'].shape} = {batch['natoms'].tolist()}")
    print(f"  batch:          {batch['batch'].shape}")
    print(f"  targets:        {batch['targets'].shape}  ← SHOULD BE [4, 12]")
    
    # Verify target shape
    expected_shape = (4, 12)  # [batch_size, num_targets]
    if batch['targets'].shape == expected_shape:
        print(f"\n✓ Target shape is correct: {batch['targets'].shape}")
    else:
        print(f"\n✗ ERROR: Target shape is {batch['targets'].shape}, expected {expected_shape}")
    
    # Show first molecule's targets
    print(f"\nFirst molecule targets (all 12 properties in PAPER ORDER):")
    property_names = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                      'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']
    for i, (name, value) in enumerate(zip(property_names, batch['targets'][0]), 1):
        print(f"  {i:2d}. {name:8s}: {value:10.4f}")
    
    print("\n" + "="*60)
    print("Data loader test complete!")
    print("="*60)