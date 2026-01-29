"""
QM9 Data Loader for PyTorch
---------------------------
This script loads the QM9 dataset using ASE and prepares PyTorch DataLoader objects
for training, validation, and testing. Assumptions:

- Molecules are isolated (no PBC)
- Target properties: all 12 QM9 targets
- Uses torch_geometric-style batching for atomic features
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
from ase import Atoms

class QM9Dataset(Dataset):
    def __init__(self, db_path, transform=None):
        """
        Args:
            db_path (str): Path to the ASE QM9 SQLite database
            transform (callable, optional): Optional transform on each sample
        """
        self.db = connect(db_path)
        self.keys = list(range(len(self.db)))  # indices for all molecules
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        Returns a dictionary with:
        - atomic_numbers: [num_atoms] atomic numbers
        - pos: [num_atoms, 3] atomic coordinates
        - natoms: number of atoms
        - targets: [12] molecular properties
        """
        row = self.db.get(self.keys[idx])
        atoms: Atoms = row.toatoms()

        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)

        # QM9 targets: μ, α, ε_HOMO, ε_LUMO, Δε, ⟨R²⟩, ZPVE, U₀, U, H, G, c_v
        targets = torch.tensor([
            row.data['mu'],
            row.data['alpha'],
            row.data['homo'],
            row.data['lumo'],
            row.data['gap'],
            row.data['r2'],
            row.data['zpve'],
            row.data['U0'],
            row.data['U'],
            row.data['H'],
            row.data['G'],
            row.data['Cv']
        ], dtype=torch.float32)

        sample = {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'natoms': len(atomic_numbers),
            'targets': targets,
            'batch': torch.zeros(len(atomic_numbers), dtype=torch.long)  # single molecule
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_qm9_loaders(db_path, batch_size=16, val_split=0.1, test_split=0.1, shuffle=True):
    """
    Returns train, val, test loaders
    """
    dataset = QM9Dataset(db_path)
    total = len(dataset)
    n_val = int(val_split * total)
    n_test = int(test_split * total)
    n_train = total - n_val - n_test

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # reproducibility
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_mol(x))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_mol(x))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_mol(x))

    return train_loader, val_loader, test_loader


def collate_mol(batch):
    """
    Collate function to handle variable number of atoms per molecule
    - Stacks atomic numbers and positions
    - Creates a batch tensor indicating molecule membership
    """
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
