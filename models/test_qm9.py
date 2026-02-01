"""
Test Script for EquiformerV2 QM9
=================================
Loads a trained model checkpoint and evaluates it on test set

Usage:
    python test_qm9.py --checkpoint path/to/best_model_checkpoint.pt
    
Or if checkpoint is in default location:
    python test_qm9.py
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.serialization.add_safe_globals([slice])  # Fix for PyTorch 2.6+
import torch.nn as nn
from tqdm import tqdm
import yaml
import json
import numpy as np

# Your modules
from data_loader_qm9_v2 import get_qm9_loaders
from equiformerv2_qm9 import EquiformerV2_QM9


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    Load model checkpoint and return model + config
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Config dictionary from checkpoint
        checkpoint: Full checkpoint dict (has epoch, losses, etc.)
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint['train_loss']:.6f}")
    print(f"  Val loss: {checkpoint['val_loss']:.6f}")
    
    # Get config
    config = checkpoint['config']
    
    # Create model with same architecture
    model_args = {
        'num_targets': config['num_targets'],
        'use_pbc': config['use_pbc'],
        'regress_forces': config['regress_forces'],
        'otf_graph': config['otf_graph'],
        'max_neighbors': config['max_neighbors'],
        'max_radius': config['cutoff_radius'],
        'max_num_elements': config['max_num_elements'],
        'num_layers': config['num_layers'],
        'sphere_channels': config['sphere_channels'],
        'attn_hidden_channels': config['attn_hidden_channels'],
        'num_heads': config['num_heads'],
        'attn_alpha_channels': config['attn_alpha_channels'],
        'attn_value_channels': config['attn_value_channels'],
        'ffn_hidden_channels': config['ffn_hidden_channels'],
        'norm_type': config['norm_type'],
        'lmax_list': config['lmax_list'],
        'mmax_list': config['mmax_list'],
        'grid_resolution': config['grid_resolution'],
        'edge_channels': config['edge_channels'],
        'use_atom_edge_embedding': config['use_atom_edge_embedding'],
        'share_atom_edge_embedding': config['share_atom_edge_embedding'],
        'use_m_share_rad': config['use_m_share_rad'],
        'distance_function': config['distance_function'],
        'num_distance_basis': config['num_radial_bases'],
        'attn_activation': config['attn_activation'],
        'use_s2_act_attn': config['use_s2_act_attn'],
        'use_attn_renorm': config['use_attn_renorm'],
        'ffn_activation': config['ffn_activation'],
        'use_gate_act': config['use_gate_act'],
        'use_grid_mlp': config['use_grid_mlp'],
        'use_sep_s2_act': config['use_sep_s2_act'],
        'alpha_drop': config['alpha_drop'],
        'drop_path_rate': config['drop_path_rate'],
        'proj_drop': config['proj_drop'],
        'weight_init': config['weight_init'],
    }
    
    model = EquiformerV2_QM9(**model_args).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    print(f"  Parameters: {model.num_params:,}")
    
    return model, config, checkpoint


def evaluate_model(model, data_loader, criterion, device='cuda'):
    """
    Evaluate model on a dataset
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            for k in batch:
                batch[k] = batch[k].to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Compute loss
            loss = criterion(predictions, batch['targets'])
            
            # Accumulate
            total_loss += loss.item() * len(batch['targets'])
            all_predictions.append(predictions.cpu())
            all_targets.append(batch['targets'].cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 12]
    all_targets = torch.cat(all_targets, dim=0)  # [N, 12]
    
    # Overall metrics
    avg_loss = total_loss / len(data_loader.dataset)
    
    # Per-property metrics
    mae_per_property = torch.abs(all_predictions - all_targets).mean(dim=0)
    mse_per_property = ((all_predictions - all_targets) ** 2).mean(dim=0)
    rmse_per_property = torch.sqrt(mse_per_property)
    
    results = {
        'overall_loss': avg_loss,
        'mae_per_property': mae_per_property.numpy(),
        'rmse_per_property': rmse_per_property.numpy(),
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy(),
    }
    
    return results


def print_results(results, property_names=None):
    """
    Print evaluation results in a nice format
    """
    if property_names is None:
        property_names = ['μ', 'α', 'ε_HOMO', 'ε_LUMO', 'Δε', '⟨R²⟩', 
                          'ZPVE', 'U₀', 'U', 'H', 'G', 'c_v']
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Loss: {results['overall_loss']:.6f}")
    
    print(f"\nPer-Property Performance:")
    print(f"{'Property':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 40)
    
    for i, name in enumerate(property_names):
        mae = results['mae_per_property'][i]
        rmse = results['rmse_per_property'][i]
        print(f"{name:<12} {mae:>11.6f} {rmse:>11.6f}")
    
    print("\n" + "="*70)


def save_results(results, save_dir, property_names=None):
    """
    Save results to files
    """
    if property_names is None:
        property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                          'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics = {
        'overall_loss': float(results['overall_loss']),
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
    
    # Save predictions and targets as numpy arrays
    np.save(os.path.join(save_dir, 'predictions.npy'), results['predictions'])
    np.save(os.path.join(save_dir, 'targets.npy'), results['targets'])
    
    print(f"\nResults saved to: {save_dir}")
    print(f"  - test_metrics.json")
    print(f"  - predictions.npy")
    print(f"  - targets.npy")


def main():
    parser = argparse.ArgumentParser(description='Test EquiformerV2 QM9 model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='trained_models/QM9/best_model_checkpoint.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='test_results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("EQUIFORMERV2 QM9 - TEST SCRIPT")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    model, config, checkpoint = load_checkpoint(args.checkpoint, args.device)
    
    # Load test data
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    
    _, _, test_loader = get_qm9_loaders(
        db_path=config['db_path'],
        batch_size=args.batch_size,
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        max_samples=config.get('max_samples', None),
        num_workers=0  # Use 0 for testing to avoid issues
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Set up loss function
    if config.get('loss_function', 'L1') == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    # Evaluate
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)
    
    results = evaluate_model(model, test_loader, criterion, args.device)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.save_dir)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE ✓")
    print("="*70)


if __name__ == '__main__':
    main()