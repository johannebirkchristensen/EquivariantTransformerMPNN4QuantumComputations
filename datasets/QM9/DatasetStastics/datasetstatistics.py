import os
import json
from ase.db import connect
import numpy as np
import torch

EV_TO_MEV = 1000 # eV to meV conversion factor.. changed to 1 to try eV units 

PROPERTY_NAMES = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                  'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']

def compute_qm9_stats(db_path,
                      max_samples=None,
                      out_dir='qm9_stats',
                      prefix='qm9',
                      print_every=5000):
    """
    Compute a suite of statistics for the 12 QM9 targets (paper units).
    Saves results to out_dir/prefix_stats.json and prefix_stats.npz, and prefix_stats.pt.

    Args:
        db_path (str): path to ASE QM9 DB (ase.db).
        max_samples (int|None): limit number of samples scanned (None -> full DB).
        out_dir (str): folder to save statistics and figures.
        prefix (str): filename prefix for saved files.
        print_every (int): progress print frequency.

    Returns:
        stats (dict): dictionary with keys per-property containing statistics.
    """
    os.makedirs(out_dir, exist_ok=True)
    db = connect(db_path)
    total = db.count()
    ids = list(range(1, total + 1))
    if max_samples is not None:
        ids = ids[:max_samples]

    n_samples = len(ids)
    # We'll collect all values into an array (n_samples, 12) to compute percentiles/median easily.
    # This uses ~ n_samples * 12 * 8 bytes (~12.5 MB for 130k samples) — fine for QM9.
    data = np.empty((n_samples, 12), dtype=np.float64)
    valid_mask = np.ones(n_samples, dtype=bool)

    for i, key in enumerate(ids, 0):
        row = db.get(id=key)
        try:
            arr = np.array([
                row.data['alpha'],
                row.data['gap'] * EV_TO_MEV,
                row.data['homo'] * EV_TO_MEV,
                row.data['lumo'] * EV_TO_MEV,
                row.data['mu'],
                row.data['Cv'],
                row.data['G'] * EV_TO_MEV,
                row.data['H'] * EV_TO_MEV,
                row.data['r2'],
                row.data['U'] * EV_TO_MEV,
                row.data['U0'] * EV_TO_MEV,
                row.data['zpve'] * EV_TO_MEV,
            ], dtype=np.float64)
        except KeyError as e:
            # If missing a property, mark invalid and continue
            print(f"Warning: missing property {e} for id {key} — marking invalid")
            valid_mask[i] = False
            data[i, :] = np.nan
            continue

        # If any NaN values, keep as NaN and mark invalid
        if np.isnan(arr).any():
            valid_mask[i] = False
        data[i, :] = arr

        if (i + 1) % print_every == 0:
            print(f"Processed {i+1}/{n_samples} samples...")

    # Restrict to valid rows for summary statistics
    good_data = data[valid_mask]
    if good_data.shape[0] == 0:
        raise RuntimeError("No valid samples found while computing QM9 stats.")

    stats = {}
    for j, name in enumerate(PROPERTY_NAMES):
        col = good_data[:, j]
        # compute stats
        count = int(np.sum(~np.isnan(col)))
        mean = float(np.nanmean(col))
        std = float(np.nanstd(col, ddof=1)) if count > 1 else 0.0
        minimum = float(np.nanmin(col))
        maximum = float(np.nanmax(col))
        median = float(np.nanmedian(col))
        p25 = float(np.nanpercentile(col, 25.0))
        p75 = float(np.nanpercentile(col, 75.0))
        nan_count = int(np.sum(np.isnan(col)))

        stats[name] = {
            'count': count,
            'mean': mean,
            'std': std,
            'min': minimum,
            'max': maximum,
            'median': median,
            'p25': p25,
            'p75': p75,
            'nan_count': nan_count,
        }

    # Save stats: JSON (human-readable) and NPZ (numeric arrays) and torch.pt
    json_path = os.path.join(out_dir, f"{prefix}_stats.json")
    npz_path = os.path.join(out_dir, f"{prefix}_stats.npz")
    pt_path = os.path.join(out_dir, f"{prefix}_stats.pt")

    # JSON: convert floats to native python floats (already are) and write
    with open(json_path, 'w') as f:
        json.dump({
            'n_samples_total': n_samples,
            'n_samples_valid': int(good_data.shape[0]),
            'property_names': PROPERTY_NAMES,
            'stats': stats
        }, f, indent=2)

    # NPZ: save raw array and valid_mask for downstream plotting/inspection
    np.savez_compressed(npz_path,
                        data=data,
                        valid_mask=valid_mask,
                        property_names=np.array(PROPERTY_NAMES, dtype=object))

    # torch.pt: save torch tensors for quick loading in training
    torch.save({
        'data': torch.from_numpy(data),
        'valid_mask': torch.from_numpy(valid_mask),
        'property_names': PROPERTY_NAMES,
        'stats': stats
    }, pt_path)

    print(f"Saved stats: {json_path}, {npz_path}, {pt_path}")

    return {
        'n_samples_total': n_samples,
        'n_samples_valid': int(good_data.shape[0]),
        'property_names': PROPERTY_NAMES,
        'stats': stats,
        'raw_npz': npz_path,
        'json': json_path,
        'pt': pt_path
    }
# ================================================================
DB_PATH = '/work3/s203788/Master_Project_2026/EquivariantTransformerMPNN4QuantumComputations/datasets/QM9/qm9_atomref_corrected.db'
result = compute_qm9_stats(DB_PATH, max_samples=None, out_dir='run_stats_corrected', prefix='qm9_full')
# result['json'] contains the path to the saved JSON summary
