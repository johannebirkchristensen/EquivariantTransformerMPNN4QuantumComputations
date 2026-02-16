from torch_geometric.datasets import QM9
from ase import Atoms
from ase.db import connect
import os

print("="*70)
print("Creating QM9 database with atomref corrections")
print("="*70)

# Load PyTorch Geometric QM9 dataset
pyg_dataset = QM9(root='./qm9_data')

# Output database path
db_path = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_atomref_corrected.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

db = connect(db_path)

# Get atomref for U₀, U, H, G (targets 7, 8, 9, 10)
print("\nLoading atomref values...")
atomref_u0 = pyg_dataset.atomref(target=7)
atomref_u = pyg_dataset.atomref(target=8)
atomref_h = pyg_dataset.atomref(target=9)
atomref_g = pyg_dataset.atomref(target=10)

print(f"\nProcessing {len(pyg_dataset)} molecules...")

for i, data in enumerate(pyg_dataset):
    atoms = Atoms(numbers=data.z.numpy(), positions=data.pos.numpy())
    
    # Calculate atomref corrections for this molecule
    correction_u0 = sum(atomref_u0[z, 0].item() for z in data.z)
    correction_u = sum(atomref_u[z, 0].item() for z in data.z)
    correction_h = sum(atomref_h[z, 0].item() for z in data.z)
    correction_g = sum(atomref_g[z, 0].item() for z in data.z)
    
    properties = {
        'mu': float(data.y[0, 0]),           # Debye
        'alpha': float(data.y[0, 1]),        # Bohr³
        'homo': float(data.y[0, 2]),         # eV
        'lumo': float(data.y[0, 3]),         # eV
        'gap': float(data.y[0, 4]),          # eV
        'r2': float(data.y[0, 5]),           # Bohr²
        'zpve': float(data.y[0, 6]),         # eV
        'U0': float(data.y[0, 7] - correction_u0),  # eV (atomref corrected)
        'U': float(data.y[0, 8] - correction_u),    # eV (atomref corrected)
        'H': float(data.y[0, 9] - correction_h),    # eV (atomref corrected)
        'G': float(data.y[0, 10] - correction_g),   # eV (atomref corrected)
        'Cv': float(data.y[0, 11]),          # cal/(mol·K)
    }
    
    db.write(atoms, data=properties)
    
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1}/{len(pyg_dataset)} molecules...")

print(f"\n✓ Database created: {db_path}")
print("\nAtomref corrections applied to: U₀, U, H, G")
print("All energies stored in eV (will be converted to meV in data loader)")