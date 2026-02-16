import os
import json
import torch
from ase.db import connect

EV_TO_MEV = 1.0

PROPERTY_ORDER = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                  'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']

DB_PATH = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9.db'
STATS_JSON = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/DatasetStastics/run_stats_eV/qm9_full_stats.json'


def load_qm9_stats_from_json(json_path):
    with open(json_path, 'r') as f:
        j = json.load(f)

    means = []
    stds = []
    for name in PROPERTY_ORDER:
        means.append(j['stats'][name]['mean'])
        stds.append(j['stats'][name]['std'])

    mean = torch.tensor(means, dtype=torch.float32)
    std = torch.tensor(stds, dtype=torch.float32)
    std[std == 0] = 1.0
    return mean, std


QM9_TARGET_MEAN, QM9_TARGET_STD = load_qm9_stats_from_json(STATS_JSON)

db = connect(DB_PATH)

print("\n=== Standardized targets for 5 molecules ===\n")

for mol_id in range(1, 6):
    row = db.get(id=mol_id)

    targets = torch.tensor([
        row.data['alpha'],
        row.data['gap']  * EV_TO_MEV,
        row.data['homo'] * EV_TO_MEV,
        row.data['lumo'] * EV_TO_MEV,
        row.data['mu'],
        row.data['Cv'],
        row.data['G']    * EV_TO_MEV,
        row.data['H']    * EV_TO_MEV,
        row.data['r2'],
        row.data['U']    * EV_TO_MEV,
        row.data['U0']   * EV_TO_MEV,
        row.data['zpve'] * EV_TO_MEV,
    ], dtype=torch.float32)

    targets_norm = (targets - QM9_TARGET_MEAN) / QM9_TARGET_STD

    print(f"Molecule {mol_id}")
    for name, val in zip(PROPERTY_ORDER, targets_norm):
        print(f"  {name:8s}: {val:8.3f}")
    print()





import torch, json
EV_TO_MEV = 1

# load mean/std (same function you used)
with open(STATS_JSON,'r') as f:
    j = json.load(f)
PROPERTY_ORDER = ['α','Δε','ε_HOMO','ε_LUMO','μ','C_v','G','H','R²','U','U₀','ZPVE']
means = torch.tensor([j['stats'][k]['mean'] for k in PROPERTY_ORDER], dtype=torch.float32)
stds  = torch.tensor([j['stats'][k]['std']  for k in PROPERTY_ORDER], dtype=torch.float32)

from ase.db import connect
db = connect(DB_PATH)

for mol_id in range(1,6):
    row = db.get(id=mol_id)
    raw = torch.tensor([
        row.data['alpha'],
        row.data['gap']*EV_TO_MEV,
        row.data['homo']*EV_TO_MEV,
        row.data['lumo']*EV_TO_MEV,
        row.data['mu'],
        row.data['Cv'],
        row.data['G']*EV_TO_MEV,
        row.data['H']*EV_TO_MEV,
        row.data['r2'],
        row.data['U']*EV_TO_MEV,
        row.data['U0']*EV_TO_MEV,
        row.data['zpve']*EV_TO_MEV,
    ], dtype=torch.float32)

    norm = (raw - means) / stds
    denorm = norm * stds + means
    print(f"mol {mol_id}")
    # check max absolute diff
    print(" max abs(diff) between raw and denorm:", float(torch.max(torch.abs(raw-denorm))))
    # print raw/G/H/U0 differences
    print(" raw G,H,U,U0 (meV):", raw[6].item(), raw[7].item(), raw[9].item(), raw[10].item())
    print()







import numpy as np
data = []
for i, rowid in enumerate(range(1, 130832)):   # or use db.count()
    row = db.get(id=rowid)
    arr = np.array([
        row.data['alpha'],
        row.data['gap']*EV_TO_MEV,
        row.data['homo']*EV_TO_MEV,
        row.data['lumo']*EV_TO_MEV,
        row.data['mu'],
        row.data['Cv'],
        row.data['G']*EV_TO_MEV,
        row.data['H']*EV_TO_MEV,
        row.data['r2'],
        row.data['U']*EV_TO_MEV,
        row.data['U0']*EV_TO_MEV,
        row.data['zpve']*EV_TO_MEV,
    ], dtype=np.float64)
    data.append(arr)
    if i>9999: break  # sample 10k for speed

data = np.stack(data, axis=0)
means = np.mean(data, axis=0)
stds  = np.std(data, axis=0, ddof=1)
print("sample means:", means)
print("sample stds: ", stds)
# now check normalization from your JSON (should match)
