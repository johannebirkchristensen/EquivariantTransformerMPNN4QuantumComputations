from ase.db import connect
import numpy as np

# Check old database
db_old = connect('/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db')
row_old = db_old.get(id=1)

# Check new database
db_new = connect('/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_atomref_corrected.db')
row_new = db_new.get(id=1)

print("First molecule comparison:")
print(f"{'Property':<10} {'Old DB (eV)':<15} {'New DB (eV)':<15} {'Difference':<15}")
print("-" * 60)

for prop in ['U0', 'U', 'H', 'G']:
    old_val = row_old.data[prop]
    new_val = row_new.data[prop]
    diff = old_val - new_val
    print(f"{prop:<10} {old_val:>14.6f} {new_val:>14.6f} {diff:>14.6f}")

print("\n" + "="*70)
print("CHECKING YOUR LOADED STATISTICS")
print("="*70)

# Load your stats file
import json
stats_path = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/DatasetStastics/run_stats_corrected/qm9_full_stats.json'

with open(stats_path, 'r') as f:
    stats = json.load(f)['stats']

print("\nStatistics for G, H, U, U₀ (in meV):")
print(f"{'Property':<10} {'Mean (meV)':<20} {'Std (meV)':<20}")
print("-" * 50)

for prop_name, idx in [('G', 'G'), ('H', 'H'), ('U', 'U'), ('U₀', 'U₀')]:
    mean = stats[prop_name]['mean']
    std = stats[prop_name]['std']
    print(f"{prop_name:<10} {mean:>19.2f} {std:>19.2f}")

print("\n" + "="*70)
print("EXPECTED VALUES (after atomref correction):")
print("="*70)
print("Mean should be around -20,000 to -100,000 meV (not millions!)")
print("Std should be around 10,000-50,000 meV")