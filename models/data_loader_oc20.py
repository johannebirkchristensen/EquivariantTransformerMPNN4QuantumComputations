"""
OC20 S2EF Data Loader
=====================
For OC20 Structure to Energy and Forces (S2EF) task
Data format: extxyz files with SinglePointCalculator
"""
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from ase.io import read
import numpy as np

# Normalization statistics for OC20 S2EF-2M (from OCP)
OC20_NORMALIZE = True

# From your config
OC20_ENERGY_MEAN = -0.7554450631141663
OC20_ENERGY_STD = 2.887317180633545
OC20_FORCE_MEAN = 0.0
OC20_FORCE_STD = 2.887317180633545


class OC20Dataset(Dataset):
    """
    OC20 S2EF Dataset from extxyz files
    
    Energy and forces are stored in atoms.calc (SinglePointCalculator)
    """
    
    def __init__(self, data_dir, normalize=True, max_samples=None):
        """
        Args:
            data_dir: Directory containing .extxyz files
            normalize: Whether to normalize energy and forces
            max_samples: Limit number of files (for debugging)
        """
        self.data_dir = data_dir
        self.normalize = normalize
        
        # Find all extxyz files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.extxyz')))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .extxyz files found in {data_dir}")
        
        if max_samples is not None:
            self.file_paths = self.file_paths[:max_samples]
        
        print(f"Found {len(self.file_paths)} .extxyz files in {data_dir}")
        
        # Count total structures
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        print("Counting structures in files...")
        for i, fpath in enumerate(self.file_paths):
            atoms_list = read(fpath, ':')
            n_structures = len(atoms_list) if isinstance(atoms_list, list) else 1
            self.file_lengths.append(n_structures)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + n_structures)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processed {i+1}/{len(self.file_paths)} files...")
        
        self.total_structures = self.cumulative_lengths[-1]
        print(f"Total structures: {self.total_structures}")
    
    def __len__(self):
        return self.total_structures
    
    def __getitem__(self, idx):
        """Get a single structure"""
        # Find which file contains this index
        file_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if self.cumulative_lengths[i] <= idx < self.cumulative_lengths[i + 1]:
                file_idx = i
                break
        
        # Get structure index within file
        structure_idx = idx - self.cumulative_lengths[file_idx]
        
        # Read the structure
        atoms = read(self.file_paths[file_idx], index=f'{structure_idx}')
        
        # Extract data
        atomic_numbers = torch.tensor(atoms.numbers, dtype=torch.long)
        pos = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
        
        # Energy and forces from calculator
        if atoms.calc is None:
            raise ValueError(f"No calculator found for structure {idx} in {self.file_paths[file_idx]}")
        
        energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float32)
        forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        
        # Fixed atoms mask (from tags: tag=0 means free, tag>0 means fixed)
        tags = atoms.get_tags()
        fixed = torch.tensor(tags > 0, dtype=torch.bool)
        
        # Normalize if requested
        if self.normalize and OC20_NORMALIZE:
            energy = (energy - OC20_ENERGY_MEAN) / OC20_ENERGY_STD
            forces = (forces - OC20_FORCE_MEAN) / OC20_FORCE_STD
        
        return {
            'atomic_numbers': atomic_numbers,
            'pos': pos,
            'cell': cell,
            'energy': energy,
            'forces': forces,
            'fixed': fixed,
            'natoms': len(atomic_numbers),
            'batch': torch.zeros(len(atomic_numbers), dtype=torch.long)
        }


def collate_oc20(batch):
    """Collate function for batching OC20 structures"""
    atomic_numbers = torch.cat([b['atomic_numbers'] for b in batch], dim=0)
    pos = torch.cat([b['pos'] for b in batch], dim=0)
    forces = torch.cat([b['forces'] for b in batch], dim=0)
    fixed = torch.cat([b['fixed'] for b in batch], dim=0)
    
    natoms = torch.tensor([b['natoms'] for b in batch], dtype=torch.long)
    batch_tensor = torch.cat([b['batch'] + i for i, b in enumerate(batch)], dim=0)
    
    energy = torch.cat([b['energy'] for b in batch], dim=0)
    
    # Stack cells into a single tensor [batch_size, 3, 3]
    cell = torch.stack([b['cell'] for b in batch], dim=0)
    
    return {
        'atomic_numbers': atomic_numbers,
        'pos': pos,
        'cell': cell,  # Now a stacked tensor instead of list
        'energy': energy,
        'forces': forces,
        'fixed': fixed,
        'natoms': natoms,
        'batch': batch_tensor
    }


def get_oc20_loaders(
    train_dir,
    val_dir,
    batch_size=64,
    num_workers=4,
    normalize=True,
    max_train_samples=None,
    max_val_samples=None
):
    """Create train and validation loaders"""
    
    train_dataset = OC20Dataset(
        train_dir,
        normalize=normalize,
        max_samples=max_train_samples
    )
    
    val_dataset = OC20Dataset(
        val_dir,
        normalize=normalize,
        max_samples=max_val_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_oc20,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_oc20,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
    )
    
    return train_loader, val_loader


def denormalize_energy(energy_normalized):
    """Convert normalized energy back to eV"""
    if not OC20_NORMALIZE:
        return energy_normalized
    return energy_normalized * OC20_ENERGY_STD + OC20_ENERGY_MEAN


def denormalize_forces(forces_normalized):
    """Convert normalized forces back to eV/Å"""
    if not OC20_NORMALIZE:
        return forces_normalized
    return forces_normalized * OC20_FORCE_STD + OC20_FORCE_MEAN
