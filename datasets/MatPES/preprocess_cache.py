#!/usr/bin/env python3
"""
preprocess_cache_matpes.py
==========================
Run this ONCE before training to pre-build the pymatgen structure caches.
This prevents the DataLoader from stalling during training when it tries to
parse + pickle 400k structures on-the-fly mid-epoch.

Usage:
    python preprocess_cache_matpes.py

The script will:
  1. Load and split the MatPES JSON (same split as training, same seed)
  2. Parse every structure via pymatgen and cache it to disk
  3. Print progress so you can see it working
  4. Exit cleanly — no training happens here

Expected runtime: 30–90 minutes depending on CPU count and disk speed.
Submit as a CPU-only job (no GPU needed), e.g.:
    #BSUB -q hpc
    #BSUB -W 02:00
    #BSUB -n 8
    #BSUB -R "rusage[mem=64GB]"
    python preprocess_cache_matpes.py
"""

import os
import sys
import time
import pickle
import gzip
import json
import warnings
from typing import Tuple

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Config (must match training config exactly) ───────────────────────────────
DATA_PATH  = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/MatPES/MatPES-PBE-2025.1.json.gz'
CACHE_DIR  = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/MatPES/cache'
TRAIN_FRAC = 0.90
VAL_FRAC   = 0.05
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pymatgen.core import Structure
except ImportError:
    raise ImportError("pip install pymatgen")


def parse_structure(struct_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(struct_dict, Structure):
            s = struct_dict          # already parsed by monty/loadfn
        else:
            s = Structure.from_dict(struct_dict)
    atomic_numbers = torch.tensor([site.specie.Z for site in s.sites], dtype=torch.long)
    pos  = torch.tensor(s.cart_coords,    dtype=torch.float32)
    cell = torch.tensor(s.lattice.matrix, dtype=torch.float32)
    return atomic_numbers, pos, cell, len(s)

def load_and_split():
    from monty.serialization import loadfn

    SPLIT_PATH = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/MatPES/MatPES-PBE-split.json.gz'

    print(f"Loading: {DATA_PATH}")
    raw     = loadfn(DATA_PATH)
    entries = list(raw) if isinstance(raw, list) else list(raw.values())
    print(f"  Raw entries: {len(entries):,}")

    print(f"Loading split indices from: {SPLIT_PATH}")
    splits    = loadfn(SPLIT_PATH)
    train_idx = set(splits["train"])
    valid_idx = set(splits["valid"])

    train_entries, val_entries, test_entries = [], [], []
    n_dropped = 0

    for i, e in enumerate(entries):
        epa = e.get('energy_per_atom')
        if epa is None:
            energy = e.get('energy')
            nsites = e.get('nsites') or len(e.get('forces') or e.get('force') or [])
            if energy is not None and nsites:
                epa = energy / nsites
        forces    = e.get('forces') or e.get('force')
        structure = e.get('structure')
        if epa is None or forces is None or structure is None:
            n_dropped += 1
            continue

        entry_slim = {'structure': structure}
        if i in train_idx:
            train_entries.append(entry_slim)
        elif i in valid_idx:
            val_entries.append(entry_slim)
        else:
            test_entries.append(entry_slim)

    print(f"  Dropped {n_dropped:,} invalid entries")
    print(f"  Split → train={len(train_entries):,}  val={len(val_entries):,}  test={len(test_entries):,}")

    return {
        'train': train_entries,
        'val':   val_entries,
        'test':  test_entries,
    }
    
    
def build_cache(entries, cache_path, split_name):
    if os.path.exists(cache_path):
        print(f"\n[{split_name}] Cache already exists at {cache_path} — skipping.")
        return

    print(f"\n[{split_name}] Building cache → {cache_path}")
    print(f"  {len(entries):,} structures to parse …")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache   = {}
    t_start = time.time()
    errors  = 0

    for idx, entry in enumerate(entries):
        try:
            cache[idx] = parse_structure(entry['structure'])
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"  WARNING idx={idx}: {exc}")
            continue

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(entries):
            elapsed = time.time() - t_start
            rate    = (idx + 1) / elapsed
            eta     = (len(entries) - idx - 1) / rate if rate > 0 else 0
            pct     = 100 * (idx + 1) / len(entries)
            print(f"  {idx+1:>7,}/{len(entries):,}  ({pct:.1f}%)  "
                  f"{rate:.0f} structs/s  ETA {eta/60:.1f} min", flush=True)

    print(f"\n  Saving pickle …")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"  ✓ Saved {len(cache):,} entries ({size_mb:.0f} MB) in {(time.time()-t_start)/60:.1f} min")
    if errors:
        print(f"  ⚠ {errors} structures failed to parse and were skipped")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    splits = load_and_split()

    for split_name, entries in splits.items():
        cache_path = os.path.join(CACHE_DIR, f'{split_name}_cache.pkl')
        build_cache(entries, cache_path, split_name)

    print("\n✓ All caches built. You can now run train_MatPESv2.py safely.")


if __name__ == '__main__':
    main()