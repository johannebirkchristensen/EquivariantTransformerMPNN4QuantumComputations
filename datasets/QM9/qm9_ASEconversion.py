from torch_geometric.datasets import QM9
from ase import Atoms
from ase.db import connect
import numpy as np

# Download QM9
pyg_dataset = QM9(root='./qm9_data')

# Create ASE database
db_path = 'qm9.db'
db = connect(db_path)

print(f"Converting {len(pyg_dataset)} molecules to ASE database...")

for i, data in enumerate(pyg_dataset):
    # Convert PyG data to ASE Atoms
    positions = data.pos.numpy()
    atomic_numbers = data.z.numpy()
    
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    
    # Add QM9 properties as key-value pairs
    # Target indices: 0=mu, 1=alpha, 2=homo, 3=lumo, 4=gap, 5=r2, 6=zpve, 7=U0, 8=U, 9=H, 10=G, 11=Cv
    properties = {
        'mu': float(data.y[0, 0]),
        'alpha': float(data.y[0, 1]),
        'homo': float(data.y[0, 2]),
        'lumo': float(data.y[0, 3]),
        'gap': float(data.y[0, 4]),
        'r2': float(data.y[0, 5]),
        'zpve': float(data.y[0, 6]),
        'U0': float(data.y[0, 7]),
        'U': float(data.y[0, 8]),
        'H': float(data.y[0, 9]),
        'G': float(data.y[0, 10]),
        'Cv': float(data.y[0, 11]),
    }
    
    db.write(atoms, data=properties)
    
    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1} molecules...")

print(f"Created ASE database at {db_path}")