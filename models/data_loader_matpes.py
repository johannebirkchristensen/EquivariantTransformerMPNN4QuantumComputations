"""
MatPES-PBE Data Loader
=======================
Supports MatPES 2025.1 JSON.gz format.

2025.1 field names:
  energy        [eV]        total DFT energy  → divided by nsites → eV/atom (target)
  forces        [eV/Å]      per-atom forces   (list of [fx, fy, fz]) (target)
  stress        [kBar]      Voigt-6 list      → converted to eV/Å³ ( target)
  structure     dict        pymatgen Structure
  nsites        int         number of atoms
  bader_magmoms list|None   per-atom magnetic moments ( target)

Requirements:
    pip install pymatgen torch numpy
"""

import os
import gzip
import json
import math
import pickle
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from pymatgen.core import Structure
except ImportError:
    raise ImportError("pip install pymatgen")

# kBar → eV/Å³
KBAR_TO_EV_ANG3 = 1.0 / 1602.1766


# ════════════════════════════════════════════════════════════════
# Low-level parsers
# ════════════════════════════════════════════════════════════════

def parse_structure(struct_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """pymatgen Structure dict → (atomic_numbers, pos_cart, cell, natoms)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = Structure.from_dict(struct_dict)
    atomic_numbers = torch.tensor([site.specie.Z for site in s.sites], dtype=torch.long)
    pos  = torch.tensor(s.cart_coords,    dtype=torch.float32)  # [N, 3] Å
    cell = torch.tensor(s.lattice.matrix, dtype=torch.float32)  # [3, 3]
    return atomic_numbers, pos, cell, len(s)


def parse_stress(stress_raw) -> torch.Tensor:
    """
    Parse stress into Voigt-6 tensor in eV/Å³.
    Accepts: None, Voigt-6 list, or 3x3 nested list. Input in kBar.
    """
    if stress_raw is None:
        return torch.zeros(6, dtype=torch.float32)
    arr = np.array(stress_raw, dtype=np.float64)
    if arr.shape == (3, 3):
        voigt = np.array([arr[0,0], arr[1,1], arr[2,2],
                          arr[1,2], arr[0,2], arr[0,1]])
    else:
        voigt = arr.flatten()[:6]
    return torch.tensor(voigt * KBAR_TO_EV_ANG3, dtype=torch.float32)


# ════════════════════════════════════════════════════════════════
# JSON loader  (handles 2025.1 field names)
# ════════════════════════════════════════════════════════════════

def load_matpes_json(data_path: str) -> list:
    """
    Load MatPES JSON.gz and normalise to canonical schema.

    2025.1 stores:
      - 'energy'  (total eV)  +  'nsites' -> we compute energy_per_atom
      - 'forces'  (not 'force')
      - 'stress'  Voigt-6 kBar
      - 'bader_magmoms' (not 'magmom')
    """
    print(f"Loading MatPES JSON from: {data_path}")
    opener = gzip.open if data_path.endswith('.gz') else open

    with opener(data_path, 'rt', encoding='utf-8') as f:
        raw = json.load(f)

    entries  = list(raw.values()) if isinstance(raw, dict) else raw
    n_before = len(entries)

    normalised = []
    for e in entries:
        # energy per atom
        epa = e.get('energy_per_atom')          # explicit field (older format)
        if epa is None:
            energy = e.get('energy')            # 2025.1: total energy
            nsites = e.get('nsites') or len(e.get('forces') or e.get('force') or [])
            if energy is not None and nsites:
                epa = energy / nsites

        # forces: 2025.1 uses 'forces', older versions used 'force'
        forces = e.get('forces') or e.get('force')

        # structure must be present
        structure = e.get('structure')

        if epa is None or forces is None or structure is None:
            continue

        normalised.append({
            'structure':       structure,
            'energy_per_atom': float(epa),
            'force':           forces,                                   # stored as 'force' internally
            'stress':          e.get('stress'),                          # Voigt-6 kBar or None
            'magmom':          e.get('bader_magmoms') or e.get('magmom'),
        })

    print(f"  {len(normalised):,} valid entries  (dropped {n_before - len(normalised):,})")
    return normalised


# ════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════

class MatPESDataset(Dataset):
    """
    MatPES PyTorch Dataset.

    Args:
        entries          : list of normalised entry dicts from load_matpes_json()
        normalize_energy : z-score energy_per_atom
        energy_mean/std  : normalization constants (pass train stats to val/test)
        normalize_stress : z-score stress (usually False)
        stress_mean/std  : per-component [6] normalization constants
        max_samples      : optional cap on number of entries
        cache_path       : path to pickle cache of parsed Structure tensors
        regress_stress   : include stress tensor in batch
        regress_magmom   : include per-atom magnetic moments in batch
    """

    def __init__(
        self,
        entries:          list,
        normalize_energy: bool            = True,
        energy_mean:      Optional[float] = None,
        energy_std:       Optional[float] = None,
        normalize_stress: bool            = False,
        stress_mean:      Optional[np.ndarray] = None,
        stress_std:       Optional[np.ndarray] = None,
        max_samples:      Optional[int]   = None,
        cache_path:       Optional[str]   = None,
        regress_stress:   bool            = True,
        regress_magmom:   bool            = False,
    ):
        self.regress_stress   = regress_stress
        self.regress_magmom   = regress_magmom
        self.normalize_energy = normalize_energy
        self.normalize_stress = normalize_stress
        self.energy_mean      = energy_mean if energy_mean is not None else 0.0
        self.energy_std       = energy_std  if energy_std  is not None else 1.0
        self.stress_mean      = stress_mean if stress_mean is not None else np.zeros(6)
        self.stress_std       = stress_std  if stress_std  is not None else np.ones(6)

        if max_samples is not None and max_samples < len(entries):
            entries = entries[:max_samples]
        self.entries = entries
        self.n       = len(entries)

        # Structure cache
        self._cache = None
        if cache_path is not None:
            if os.path.exists(cache_path):
                print(f"  Loading cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"  Cache: {len(self._cache):,} entries")
            else:
                print(f"  Cache not found -- will parse on-the-fly -> {cache_path}")
                self._cache = {}
                self._cache_save_path = cache_path

    def compute_energy_stats(self) -> Tuple[float, float]:
        vals = np.array([e['energy_per_atom'] for e in self.entries], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    def compute_stress_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.stack([parse_stress(e.get('stress')).numpy() for e in self.entries])
        return arr.mean(axis=0), arr.std(axis=0)

    def save_cache(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        print(f"Cache saved: {path}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        entry = self.entries[idx]

        # Structure (cached)
        if self._cache is not None and idx in self._cache:
            atomic_numbers, pos, cell, natoms = self._cache[idx]
        else:
            atomic_numbers, pos, cell, natoms = parse_structure(entry['structure'])
            if self._cache is not None:
                self._cache[idx] = (atomic_numbers, pos, cell, natoms)

        # Energy
        energy = torch.tensor([entry['energy_per_atom']], dtype=torch.float32)
        if self.normalize_energy:
            energy = (energy - self.energy_mean) / self.energy_std

        # Forces [N, 3] eV/Å -- NOT normalised
        forces = torch.tensor(entry['force'], dtype=torch.float32)

        # Stress [6] eV/Å³
        stress = parse_stress(entry.get('stress'))
        if self.normalize_stress:
            std_safe = np.where(self.stress_std > 1e-8, self.stress_std, 1.0)
            stress = (stress - torch.tensor(self.stress_mean, dtype=torch.float32)) \
                     / torch.tensor(std_safe, dtype=torch.float32)

        out = {
            'atomic_numbers': atomic_numbers,                          # [N]
            'pos':            pos,                                      # [N, 3]
            'cell':           cell,                                     # [3, 3]
            'energy':         energy,                                   # [1]
            'forces':         forces,                                   # [N, 3]
            'natoms':         natoms,                                   # int
            'batch':          torch.zeros(natoms, dtype=torch.long),   # [N]
        }
        if self.regress_stress:
            out['stress'] = stress                                      # [6]
        if self.regress_magmom:
            mg = entry.get('magmom')
            out['magmom'] = torch.tensor(mg, dtype=torch.float32) \
                            if mg is not None \
                            else torch.zeros(natoms, dtype=torch.float32)
        return out


# ════════════════════════════════════════════════════════════════
# Collate
# ════════════════════════════════════════════════════════════════

def collate_matpes(batch):
    atomic_numbers = torch.cat([b['atomic_numbers'] for b in batch])
    pos            = torch.cat([b['pos']            for b in batch])
    forces         = torch.cat([b['forces']         for b in batch])
    energy         = torch.cat([b['energy']         for b in batch])
    natoms         = torch.tensor([b['natoms'] for b in batch], dtype=torch.long)
    batch_tensor   = torch.cat([b['batch'] + i for i, b in enumerate(batch)])
    cell           = torch.stack([b['cell'] for b in batch])            # [B, 3, 3]
    pbc            = torch.tensor([[True, True, True]] * len(batch), dtype=torch.bool)

    out = {
        'atomic_numbers': atomic_numbers,   # [total_atoms]
        'pos':            pos,              # [total_atoms, 3]
        'cell':           cell,             # [B, 3, 3]
        'energy':         energy,           # [B, 1]
        'forces':         forces,           # [total_atoms, 3]
        'natoms':         natoms,           # [B]
        'batch':          batch_tensor,     # [total_atoms]
        'pbc':            pbc,              # [B, 3]
    }
    if 'stress' in batch[0]:
        out['stress'] = torch.stack([b['stress'] for b in batch])       # [B, 6]
    if 'magmom' in batch[0]:
        out['magmom'] = torch.cat([b['magmom'] for b in batch])         # [total_atoms]
    return out


# ════════════════════════════════════════════════════════════════
# Loader factory
# ════════════════════════════════════════════════════════════════

def get_matpes_loaders(
    data_path:        str,
    batch_size:       int           = 16,
    num_workers:      int           = 4,
    train_frac:       float         = 0.90,
    val_frac:         float         = 0.05,
    normalize_energy: bool          = True,
    normalize_stress: bool          = False,
    max_train:        Optional[int] = None,
    max_val:          Optional[int] = None,
    max_test:         Optional[int] = None,
    cache_dir:        Optional[str] = None,
    regress_stress:   bool          = True,
    regress_magmom:   bool          = False,
    random_seed:      int           = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders for MatPES."""

    all_entries = load_matpes_json(data_path)

    rng     = np.random.default_rng(random_seed)
    idx     = rng.permutation(len(all_entries)).tolist()
    n       = len(idx)
    n_train = int(train_frac * n)
    n_val   = int(val_frac   * n)

    train_entries = [all_entries[i] for i in idx[:n_train]]
    val_entries   = [all_entries[i] for i in idx[n_train : n_train + n_val]]
    test_entries  = [all_entries[i] for i in idx[n_train + n_val :]]

    print(f"\nSplit  train={len(train_entries):,}  val={len(val_entries):,}  test={len(test_entries):,}")

    def _cache_path(split):
        if cache_dir is None:
            return None
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'{split}_cache.pkl')

    # Train (compute stats before normalising)
    print("\n-- TRAIN --")
    train_ds = MatPESDataset(
        train_entries, normalize_energy=False,
        max_samples=max_train, cache_path=_cache_path('train'),
        regress_stress=regress_stress, regress_magmom=regress_magmom,
    )

    print("  Computing energy statistics...")
    energy_mean, energy_std = train_ds.compute_energy_stats()
    print(f"  Energy  mean={energy_mean:.6f}  std={energy_std:.6f}  eV/atom")

    stress_mean, stress_std = np.zeros(6), np.ones(6)
    if normalize_stress and regress_stress:
        print("  Computing stress statistics...")
        stress_mean, stress_std = train_ds.compute_stress_stats()

    # Apply stats to train
    train_ds.normalize_energy = normalize_energy
    train_ds.energy_mean      = energy_mean
    train_ds.energy_std       = energy_std
    train_ds.normalize_stress = normalize_stress
    train_ds.stress_mean      = stress_mean
    train_ds.stress_std       = stress_std

    print("\n-- VAL --")
    val_ds = MatPESDataset(
        val_entries,
        normalize_energy=normalize_energy, energy_mean=energy_mean, energy_std=energy_std,
        normalize_stress=normalize_stress, stress_mean=stress_mean, stress_std=stress_std,
        max_samples=max_val, cache_path=_cache_path('val'),
        regress_stress=regress_stress, regress_magmom=regress_magmom,
    )

    print("\n-- TEST --")
    test_ds = MatPESDataset(
        test_entries,
        normalize_energy=normalize_energy, energy_mean=energy_mean, energy_std=energy_std,
        normalize_stress=normalize_stress, stress_mean=stress_mean, stress_std=stress_std,
        max_samples=max_test, cache_path=_cache_path('test'),
        regress_stress=regress_stress, regress_magmom=regress_magmom,
    )

    kw = dict(
        collate_fn=collate_matpes,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    print(f"\nLoaders ready:")
    print(f"  Train: {len(train_ds):>7,}  ({math.ceil(len(train_ds)/batch_size)} batches)")
    print(f"  Val  : {len(val_ds):>7,}  ({math.ceil(len(val_ds)/batch_size)} batches)")
    print(f"  Test : {len(test_ds):>7,}  ({math.ceil(len(test_ds)/batch_size)} batches)")

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def denormalize_energy(energy_norm, mean, std):
    return energy_norm * std + mean


# ════════════════════════════════════════════════════════════════
# Smoke test
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   required=True)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_train',   type=int, default=100)
    args = parser.parse_args()

    tl, vl, tel = get_matpes_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train=args.max_train,
        max_val=20,
        max_test=20,
    )

    print("\n-- Smoke test: first train batch --")
    batch = next(iter(tl))
    for k, v in batch.items():
        print(f"  {k:20s}: shape={v.shape}  dtype={v.dtype}")
    print("\n  OK - MatPES data loader smoke test PASSED")