#!/usr/bin/env python3
"""
Test script for EquiformerV2_MatPES
-------------------------------------
Loads a trained checkpoint and evaluates on the test set.
Reports Energy, Force, and Stress MAEs in physical units.
"""
import torch
try:
    torch.serialization.add_safe_globals([slice])
except Exception:
    pass

import os, sys, json, argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader_matpes import get_matpes_loaders
from equiformerv2_MatPESv2 import EquiformerV2_MatPES

EV_ANG3_TO_GPA = 160.2176621


# ════════════════════════════════════════════════════════════════
# Load checkpoint
# ════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg  = ckpt['config']
    best_f_mae = ckpt.get('best_val_f_mae', None)
    best_f_str = f"{best_f_mae:.3f}" if best_f_mae is not None else "?"
    print(f"  Epoch: {ckpt.get('epoch', '?')}   Best Force MAE: {best_f_str} meV/Å")

    model = EquiformerV2_MatPES(
        use_pbc=cfg.get('use_pbc', True),
        regress_forces=cfg.get('regress_forces', True),
        regress_stress=cfg.get('regress_stress', True),
        max_neighbors=cfg.get('max_neighbors', 20),
        max_radius=cfg.get('cutoff_radius', 6.0),
        max_num_elements=cfg.get('max_num_elements', 100),
        num_layers=cfg.get('num_layers', 6),
        sphere_channels=cfg.get('sphere_channels', 128),
        attn_hidden_channels=cfg.get('attn_hidden_channels', 128),
        num_heads=cfg.get('num_heads', 8),
        attn_alpha_channels=cfg.get('attn_alpha_channels', 32),
        attn_value_channels=cfg.get('attn_value_channels', 16),
        ffn_hidden_channels=cfg.get('ffn_hidden_channels', 512),
        norm_type=cfg.get('norm_type', 'rms_norm_sh'),
        lmax_list=cfg.get('lmax_list', [4]),
        mmax_list=cfg.get('mmax_list', [2]),
        grid_resolution=cfg.get('grid_resolution', 18),
        edge_channels=cfg.get('edge_channels', 128),
        use_atom_edge_embedding=cfg.get('use_atom_edge_embedding', True),
        share_atom_edge_embedding=cfg.get('share_atom_edge_embedding', False),
        use_m_share_rad=cfg.get('use_m_share_rad', False),
        distance_function=cfg.get('distance_function', 'gaussian'),
        num_distance_basis=cfg.get('num_radial_bases', 512),
        attn_activation=cfg.get('attn_activation', 'scaled_silu'),
        use_s2_act_attn=cfg.get('use_s2_act_attn', False),
        use_attn_renorm=cfg.get('use_attn_renorm', True),
        ffn_activation=cfg.get('ffn_activation', 'scaled_silu'),
        use_gate_act=cfg.get('use_gate_act', False),
        use_grid_mlp=cfg.get('use_grid_mlp', False),
        use_sep_s2_act=cfg.get('use_sep_s2_act', True),
        alpha_drop=0.0,        # no dropout at test time
        drop_path_rate=0.0,
        proj_drop=0.0,
        weight_init=cfg.get('weight_init', 'normal'),
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Parameters: {model.num_params:,}")

    energy_mean = ckpt.get('energy_mean', 0.0)
    energy_std  = ckpt.get('energy_std',  1.0)
    regress_stress = cfg.get('regress_stress', True)
    return model, cfg, energy_mean, energy_std, regress_stress


# ════════════════════════════════════════════════════════════════
# Evaluate
# ════════════════════════════════════════════════════════════════
def evaluate(model, loader, device, energy_std, regress_stress, use_amp=False):
    model.eval()

    all_e_err, all_f_err, all_s_err = [], [], []
    all_f_ratio = []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forces require grad on pos
        batch['pos'].requires_grad_(True)

        out = model(batch)

        # Compute forces via autograd: F = -dE_total/dr
        energy_total = out['energy_total']   # [B]
        forces = -torch.autograd.grad(
            energy_total.sum(),
            batch['pos'],
            create_graph=False,
            retain_graph=False,
        )[0]   # [N, 3]

        # Energy
        e_pred = out['energy'].flatten()
        e_tgt  = batch['energy'].flatten()
        e_err  = (e_pred - e_tgt).abs().detach().cpu().numpy() * energy_std * 1000
        all_e_err.extend(e_err.tolist())

        # Forces MAE
        f_tgt = batch['forces']
        f_err = (forces - f_tgt).abs().mean(dim=1).detach().cpu().numpy() * 1000
        all_f_err.extend(f_err.tolist())

        # f/f_DFT
        f_pred_norm = torch.norm(forces, dim=1)
        f_tgt_norm  = torch.norm(f_tgt,  dim=1)
        ratio = (f_pred_norm / f_tgt_norm.clamp(min=1e-3)).detach().cpu().numpy()
        all_f_ratio.extend(ratio.tolist())

        # Stress
        if regress_stress and 'stress' in out and 'stress' in batch:
            s_pred = out['stress']
            s_tgt  = batch['stress']
            s_err  = (s_pred - s_tgt).abs().mean(dim=1).detach().cpu().numpy() * EV_ANG3_TO_GPA
            all_s_err.extend(s_err.tolist())

    results = {
        'energy_mae_meV':    float(np.mean(all_e_err)),
        'energy_std_meV':    float(np.std(all_e_err)),
        'force_mae_meV_ang': float(np.mean(all_f_err)),
        'force_std_meV_ang': float(np.std(all_f_err)),
        'force_ratio':       float(np.mean(all_f_ratio)),
        'force_ratio_std':   float(np.std(all_f_ratio)),
    }
    if all_s_err:
        results['stress_mae_GPa'] = float(np.mean(all_s_err))
        results['stress_std_GPa'] = float(np.std(all_s_err))

    return results


# ════════════════════════════════════════════════════════════════
# Print
# ════════════════════════════════════════════════════════════════

def print_results(results, split='Test'):
    # Energy: Ef MAE in eV/atom from MatCalc-Benchmark
    # Forces: f/f_DFT ratio (1.0 = perfect)
    m3gnet    = {'energy': 0.11,  'f_ratio': 0.97}
    chgnet    = {'energy': 0.082, 'f_ratio': 0.91}
    tensornet = {'energy': 0.081, 'f_ratio': 0.93}

    print(f"\n{'='*85}")
    print(f"{split.upper()} RESULTS")
    print(f"{'='*85}")
    print(f"\n{'Property':<22} {'Ours':>10} {'Std':>10} {'M3GNet':>10} {'CHGNet':>10} {'TensorNet':>12}")
    print("-" * 76)
    print(f"{'Energy (meV/atom)':<22} {results['energy_mae_meV']:>10.3f}"
          f" {results.get('energy_std_meV', 0):>10.3f}"
          f" {m3gnet['energy']*1000:>10.1f}"
          f" {chgnet['energy']*1000:>10.1f}"
          f" {tensornet['energy']*1000:>12.1f}")
    if 'force_ratio' in results:
        print(f"{'Forces (f/f_DFT)':<22} {results['force_ratio']:>10.3f}"
              f" {'':>10}"
              f" {m3gnet['f_ratio']:>10.2f}"
              f" {chgnet['f_ratio']:>10.2f}"
              f" {tensornet['f_ratio']:>12.2f}")
    print(f"{'Forces (meV/Å)':<22} {results['force_mae_meV_ang']:>10.3f}"
          f" {results.get('force_std_meV_ang', 0):>10.3f}"
          f"{'  (no benchmark — metric differs)':>34}")
    print(f"\n{'='*85}")
# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Evaluate EquiformerV2 MatPES')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model_checkpoint.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Where to save results JSON (default: checkpoint dir)')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--max_test', type=int, default=None,
                        help='Cap on test samples (for quick testing)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model, cfg, energy_mean, energy_std, regress_stress = load_model(
        args.checkpoint, device
    )

    # Reconstruct test loader using the same split logic
    data_path = cfg['data_path']
    print(f"\nLoading test data from: {data_path}")
    _, _, test_loader = get_matpes_loaders(
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_frac=cfg.get('train_frac', 0.90),
        val_frac=cfg.get('val_frac', 0.05),
        normalize_energy=True,
        max_test=args.max_test,
        regress_stress=regress_stress,
        random_seed=cfg.get('random_seed', 42),
    )
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"\n  Checkpoint energy_std : {energy_std:.6f} eV/atom")
    print(f"  Recomputed energy_std : {test_loader.dataset.energy_std:.6f} eV/atom")
    if abs(energy_std - test_loader.dataset.energy_std) > 0.01:
        print("  WARNING: energy_std mismatch — results may be wrong!")
    results = evaluate(
        model, test_loader, device,
        energy_std=energy_std,
        regress_stress=regress_stress,
        use_amp=args.use_amp,
    )

    print_results(results, split='Test')

    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    out_path = os.path.join(save_dir, 'test_metrics_matpes.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()