import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BackboneInterface as BaseModel

# Updated paths for sampling and smearing
from fairchem.core.models.uma.common.sampling import CalcSpherePoints
from fairchem.core.models.escaip.utils.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass
    logging.warning("e3nn not found. Please install e3nn to use EquiformerV2_OC20 model.")
    
    
    
from fairchem.core.datasets import AseReadDataset
from ase.db import connect

# Download QM9 database (if you don't have it)
# You can get it from: https://doi.org/10.6084/m9.figshare.978904

# Create ASE database from QM9
# Assuming you have qm9.db file
#dataset = AseReadDataset(config={
 #   'src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db',  # or qm9.xyz file
 #   'target': 'U0',  # or whatever property you want
#})






print("Loading QM9 dataset from ASE database...")

from fairchem.core.datasets import AseDBDataset

dataset = AseDBDataset(config={
    'src': '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_small_1000.db',
    'target': 'U0',
})

print(f"Loaded {len(dataset)} samples of QM9 dataset from ASE database")

# Access just one sample to test
print("First sample:")
print(dataset[0])