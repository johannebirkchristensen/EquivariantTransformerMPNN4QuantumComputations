"""
Test Script for EquiformerV2 QM9 - CORRECTED
=============================================
Properly handles:
- Target denormalization
- Correct property order
- MAE computation in paper units
"""

import sys
import os
import argparse

import torch
torch.serialization.add_safe_globals([slice])
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np

from data_loader_qm9_v3 import (
    get_qm9_loaders,
    denormalize_targets,
    QM9_TARGET_MEAN,
    QM9_TARGET_STD,
    QM9_NORMALIZE
)
from equiformerv2_qm9 import EquiformerV2_QM9


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load checkpoint and create model"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.6f}")
    print(f"  Val loss:   {checkpoint['val_loss']:.6f}")
    
    config = checkpoint['config']
    print("\nModel Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model with same architecture
    model_args = {
        'num_targets': config['num_targets'],
        'use_pbc': config.get('use_pbc', False),
        'regress_forces': config.get('regress_forces', False),
        'otf_graph': config.get('otf_graph', True),
        'max_neighbors': config.get('max_neighbors', 500),
        'max_radius': config.get('cutoff_radius', 5.0),
        'max_num_elements': config.get('max_num_elements', 10),
        'num_layers': config.get('num_layers', 6),
        'sphere_channels': config.get('sphere_channels', 96),
        'attn_hidden_channels': config.get('attn_hidden_channels', 48),
        'num_heads': config.get('num_heads', 4),
        'attn_alpha_channels': config.get('attn_alpha_channels', 64),
        'attn_value_channels': config.get('attn_value_channels', 24),
        'ffn_hidden_channels': config.get('ffn_hidden_channels', 96),
        'norm_type': config.get('norm_type', 'rms_norm_sh'),
        'lmax_list': config.get('lmax_list', [4]),
        'mmax_list': config.get('mmax_list', [4]),
        'grid_resolution': config.get('grid_resolution', 18),
        'edge_channels': config.get('edge_channels', 64),
        'use_atom_edge_embedding': config.get('use_atom_edge_embedding', True),
        'share_atom_edge_embedding': config.get('share_atom_edge_embedding', False),
        'use_m_share_rad': config.get('use_m_share_rad', False),
        'distance_function': config.get('distance_function', 'gaussian'),
        'num_distance_basis': config.get('num_radial_bases', 128),
        'attn_activation': config.get('attn_activation', 'scaled_silu'),
        'use_s2_act_attn': config.get('use_s2_act_attn', False),
        'use_attn_renorm': config.get('use_attn_renorm', True),
        'ffn_activation': config.get('ffn_activation', 'scaled_silu'),
        'use_gate_act': config.get('use_gate_act', False),
        'use_grid_mlp': config.get('use_grid_mlp', False),
        'use_sep_s2_act': config.get('use_sep_s2_act', True),
        'alpha_drop': config.get('alpha_drop', 0.1),
        'drop_path_rate': config.get('drop_path_rate', 0.05),
        'proj_drop': config.get('proj_drop', 0.0),
        'weight_init': config.get('weight_init', 'normal'),
    }
    
    model = EquiformerV2_QM9(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {model.num_params:,} parameters")
    
    return model, config, checkpoint


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate model and compute MAEs in PAPER UNITS (denormalized)
    """
    model.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Forward pass (outputs are normalized if training was normalized)
            predictions_norm = model(batch)
            targets_norm = batch['targets']
            
            all_preds_norm.append(predictions_norm.cpu())
            all_targets_norm.append(targets_norm.cpu())
    
    # Concatenate
    all_preds_norm = torch.cat(all_preds_norm, dim=0)  # [N, 12]
    all_targets_norm = torch.cat(all_targets_norm, dim=0)  # [N, 12]
    
    # ================================================================
    # DENORMALIZE to get MAE in paper units
    # ================================================================
    if QM9_NORMALIZE:
        all_preds = denormalize_targets(all_preds_norm)
        all_targets = denormalize_targets(all_targets_norm)
    else:
        all_preds = all_preds_norm
        all_targets = all_targets_norm
    
    # Compute MAE in PAPER UNITS
    mae_per_property = torch.abs(all_preds - all_targets).mean(dim=0)
    rmse_per_property = torch.sqrt(((all_preds - all_targets) ** 2).mean(dim=0))
    
    # Overall MAE (across all properties and samples)
    overall_mae = torch.abs(all_preds - all_targets).mean()
    
    results = {
        'overall_mae': overall_mae.item(),
        'mae_per_property': mae_per_property.numpy(),
        'rmse_per_property': rmse_per_property.numpy(),
        'predictions': all_preds.numpy(),
        'targets': all_targets.numpy(),
    }
    
    return results


def print_results(results):
    """Print results in paper format"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS (in PAPER UNITS)")
    print("="*70)
    
    print(f"\nOverall MAE: {results['overall_mae']:.6f}")
    
    print(f"\nPer-Property MAE (Paper Order):")
    print(f"{'Property':<12} {'MAE':<12} {'RMSE':<12} {'Paper MAE':<12}")
    print("-" * 55)
    
    # Paper order with units and paper MAEs for comparison
    properties = [
        ('α (Bohr³)',    'α',      0.050),
        ('Δε (meV)',     'Δε',     29),
        ('ε_HOMO (meV)', 'ε_HOMO', 14),
        ('ε_LUMO (meV)', 'ε_LUMO', 13),
        ('μ (Debye)',    'μ',      0.010),
        ('C_v',          'C_v',    0.023),
        ('G (meV)',      'G',      7.57),
        ('H (meV)',      'H',      6.22),
        ('R² (Bohr²)',   'R²',     0.186),
        ('U (meV)',      'U',      6.49),
        ('U₀ (meV)',     'U₀',     6.17),
        ('ZPVE (meV)',   'ZPVE',   1.47),
    ]
    
    for i, (name, symbol, paper_mae) in enumerate(properties):
        mae = results['mae_per_property'][i]
        rmse = results['rmse_per_property'][i]
        print(f"{name:<12} {mae:>11.6f} {rmse:>11.6f} {paper_mae:>11.6f}")
    
    print("\n" + "="*70)


def save_results(results, save_dir):
    """Save results to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics
    property_names = ['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv',
                      'G', 'H', 'r2', 'U', 'U0', 'zpve']
    
    metrics = {
        'overall_mae': float(results['overall_mae']),
        'per_property_mae': {
            name: float(mae)
            for name, mae in zip(property_names, results['mae_per_property'])
        },
        'per_property_rmse': {
            name: float(rmse)
            for name, rmse in zip(property_names, results['rmse_per_property'])
        },
    }
    
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions and targets (in paper units)
    np.save(os.path.join(save_dir, 'predictions.npy'), results['predictions'])
    np.save(os.path.join(save_dir, 'targets.npy'), results['targets'])
    
    print(f"\nResults saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='trained_models/QM9/20260202_090709/best_model_checkpoint.pt',
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='test_results',
        help='Save directory'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EQUIFORMERV2 QM9 - CORRECTED TEST SCRIPT")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    model, config, checkpoint = load_checkpoint(args.checkpoint, args.device)
    
    # Load test data WITH SAME NORMALIZATION AS TRAINING
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    _, _, test_loader = get_qm9_loaders(
        db_path=config['db_path'],
        batch_size=args.batch_size,
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        max_samples=config.get('max_samples', None),
        num_workers=0,
        normalize=QM9_NORMALIZE  # Use same normalization as training
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    results = evaluate_model(model, test_loader, args.device)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.save_dir)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE ✓")
    print("="*70)


if __name__ == '__main__':
    main()