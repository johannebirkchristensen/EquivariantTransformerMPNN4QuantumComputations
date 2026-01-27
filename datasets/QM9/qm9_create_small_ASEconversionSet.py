
###########################
    
from fairchem.core.datasets import AseReadDataset


from ase.db import connect


print("Creating small test database...")

# Read from original
db_src = connect('/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db')

# Create smaller database with just 1000 samples
db_small = connect('/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_small.db')

for i, row in enumerate(db_src.select()):
    if i >= 1000:  # Change this number as needed
        break
    atoms = row.toatoms()
    db_small.write(atoms, data=row.data)
    if (i + 1) % 100 == 0:
        print(f"Copied {i + 1} molecules...")

print("✓ Created qm9_small.db with 1000 samples")

#################################3