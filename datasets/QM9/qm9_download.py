from torch_geometric.datasets import QM9
import os

# This will download QM9 automatically
dataset = QM9(root='./qm9_data')
print(f"Downloaded {len(dataset)} molecules")


