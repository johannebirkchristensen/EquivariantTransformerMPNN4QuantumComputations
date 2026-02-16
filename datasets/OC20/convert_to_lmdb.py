#!/usr/bin/env python3
"""
Convert OC20 S2EF extxyz files to LMDB format
==============================================
Takes ~3-4 hours for S2EF-2M (~2M structures)
Final size: ~80GB per split
"""
import lmdb
import pickle
from ase.io import read
from tqdm import tqdm
import os
import glob
import sys
import numpy as np


def convert_to_lmdb(src_dir, lmdb_path, map_size=150e9):
    """
    Convert extxyz files to LMDB database
    
    Args:
        src_dir: Directory containing .extxyz files
        lmdb_path: Output LMDB file path (will create this file)
        map_size: Maximum database size in bytes (150GB default)
    """
    
    print(f"\n{'='*60}")
    print(f"Converting: {src_dir}")
    print(f"Output: {lmdb_path}")
    print(f"Map size: {map_size/1e9:.1f} GB")
    print(f"{'='*60}\n")
    
    # Find all extxyz files
    extxyz_files = sorted(glob.glob(os.path.join(src_dir, '*.extxyz')))
    
    if len(extxyz_files) == 0:
        raise ValueError(f"No .extxyz files found in {src_dir}")
    
    print(f"Found {len(extxyz_files)} .extxyz files")
    
    # Create LMDB environment
    env = lmdb.open(
        lmdb_path,
        map_size=int(map_size),
        subdir=False,           # Single file, not directory
        meminit=False,          # Don't initialize memory
        map_async=True,         # Async writes for speed
    )
    
    structure_count = 0
    txn = env.begin(write=True)
    
    # Process each file
    for file_idx, fpath in enumerate(extxyz_files):
        print(f"\nProcessing file {file_idx+1}/{len(extxyz_files)}: {os.path.basename(fpath)}")
        
        try:
            # Read all structures from file
            atoms_list = read(fpath, ':')
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            
            print(f"  Found {len(atoms_list)} structures")
            
            # Store each structure
            for atoms in tqdm(atoms_list, desc="  Storing", leave=False):
                # Extract data
                data = {
                    'atomic_numbers': np.array(atoms.numbers, dtype=np.int32),
                    'pos': np.array(atoms.positions, dtype=np.float32),
                    'cell': np.array(atoms.cell.array, dtype=np.float32),
                    'energy': float(atoms.get_potential_energy()),
                    'forces': np.array(atoms.get_forces(), dtype=np.float32),
                    'tags': np.array(atoms.get_tags(), dtype=np.int32),
                    'natoms': len(atoms),
                }
                
                # Store in LMDB
                key = f'{structure_count}'.encode('ascii')
                txn.put(key, pickle.dumps(data, protocol=4))
                
                structure_count += 1
                
                # Commit periodically to avoid memory issues
                if structure_count % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        
        except Exception as e:
            print(f"  ERROR reading {fpath}: {e}")
            continue
    
    # Store total count
    txn.put(b'length', str(structure_count).encode('ascii'))
    txn.commit()
    
    # Close database
    env.close()
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"Total structures: {structure_count:,}")
    print(f"Output: {lmdb_path}")
    
    # Check file size
    if os.path.exists(lmdb_path):
        size_gb = os.path.getsize(lmdb_path) / 1e9
        print(f"File size: {size_gb:.2f} GB")
    print(f"{'='*60}\n")


def main():
    """Convert both training and validation datasets"""
    
    base_path = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/OC20/data/s2ef'
    
    # Training data
    print("\n" + "="*70)
    print("CONVERTING TRAINING DATA (S2EF-2M)")
    print("="*70)
    
    train_src = os.path.join(base_path, 'raw/s2ef_train_2M/s2ef_train_2M')
    train_lmdb = os.path.join(base_path, 'lmdb/s2ef_train_2M.lmdb')
    
    os.makedirs(os.path.dirname(train_lmdb), exist_ok=True)
    
    convert_to_lmdb(
        src_dir=train_src,
        lmdb_path=train_lmdb,
        map_size=150e9  # 150GB (training is large)
    )
    
    # Validation data
    print("\n" + "="*70)
    print("CONVERTING VALIDATION DATA")
    print("="*70)
    
    val_src = os.path.join(base_path, 'raw/s2ef_val_id_uncompressed')
    val_lmdb = os.path.join(base_path, 'lmdb/s2ef_val_id.lmdb')
    
    convert_to_lmdb(
        src_dir=val_src,
        lmdb_path=val_lmdb,
        map_size=20e9  # 20GB (validation is smaller)
    )
    
    print("\n" + "="*70)
    print("✓ ALL CONVERSIONS COMPLETE!")
    print("="*70)
    print("\nYou can now delete the .extxyz files to save space:")
    print(f"  rm -rf {train_src}")
    print(f"  rm -rf {val_src}")
    print("\nThis will free up ~71GB of space.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()