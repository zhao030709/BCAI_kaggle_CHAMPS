"""
Evaluation script for trained Set Transformer model.
"""
import torch
import numpy as np
import pandas as pd
from set_transformer_model import create_model
from set_transformer_data import MolecularGraphDataset, collate_fn
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
import json
import os


def evaluate_model(model_path, data_csv, device='cpu', batch_size=8):
    """
    Evaluate trained Set Transformer model.
    
    Args:
        model_path: Path to saved checkpoint
        data_csv: Path to pseudo-labeled dataset
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = create_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load data
    from set_transformer_data import load_pseudo_labels
    smiles_list, coupling_data = load_pseudo_labels(data_csv)
    
    dataset = MolecularGraphDataset(
        smiles_list, coupling_data,
        max_atoms=config['model']['max_atoms']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Evaluation metrics
    all_metrics = {
        'j_mae': [],
        'j_rmse': [],
        'type_accuracy': [],
        'matched_pairs': [],
        'total_targets': [],
        'matching_rate': []
    }
    
    predictions = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            
            # Get predictions
            pred_j = output['j_values']  # [batch, n_pairs]
            pred_type_logits = output['type_logits']  # [batch, n_pairs, n_types]
            pred_types = torch.argmax(pred_type_logits, dim=-1)  # [batch, n_pairs]
            pred_mask = batch['pred_mask']  # [batch, n_pairs]
            
            target_j = batch['target_j']  # [batch, max_targets]
            target_types = batch['target_types']  # [batch, max_targets]
            target_mask = batch['target_mask']  # [batch, max_targets]
            
            # Process each sample in batch
            for i in range(len(pred_j)):
                # Get valid predictions and targets
                valid_pred_idx = pred_mask[i].cpu().numpy()
                valid_target_idx = target_mask[i].cpu().numpy()
                
                pred_j_valid = pred_j[i][valid_pred_idx].cpu().numpy()
                pred_types_valid = pred_types[i][valid_pred_idx].cpu().numpy()
                
                target_j_valid = target_j[i][valid_target_idx].cpu().numpy()
                target_types_valid = target_types[i][valid_target_idx].cpu().numpy()
                
                # Hungarian matching
                n_pred = len(pred_j_valid)
                n_target = len(target_j_valid)
                
                if n_pred == 0 or n_target == 0:
                    continue
                
                # Create cost matrix (MAE for J-values)
                cost_matrix = np.abs(pred_j_valid[:, None] - target_j_valid[None, :])
                
                # Solve assignment problem
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Calculate metrics on matched pairs
                matched_pred_j = pred_j_valid[row_ind]
                matched_target_j = target_j_valid[col_ind]
                matched_pred_types = pred_types_valid[row_ind]
                matched_target_types = target_types_valid[col_ind]
                
                j_errors = np.abs(matched_pred_j - matched_target_j)
                j_mae = np.mean(j_errors)
                j_rmse = np.sqrt(np.mean(j_errors ** 2))
                type_acc = np.mean(matched_pred_types == matched_target_types)
                
                all_metrics['j_mae'].append(j_mae)
                all_metrics['j_rmse'].append(j_rmse)
                all_metrics['type_accuracy'].append(type_acc)
                all_metrics['matched_pairs'].append(len(row_ind))
                all_metrics['total_targets'].append(n_target)
                all_metrics['matching_rate'].append(len(row_ind) / n_target)
                
                # Store predictions
                for matched_idx, (pred_idx, target_idx) in enumerate(zip(row_ind, col_ind)):
                    predictions.append({
                        'smiles': smiles_list[batch_idx * batch_size + i],
                        'pred_j': float(pred_j_valid[pred_idx]),
                        'target_j': float(target_j_valid[target_idx]),
                        'pred_type': int(pred_types_valid[pred_idx]),
                        'target_type': int(target_types_valid[target_idx]),
                        'error': float(j_errors[matched_idx])
                    })
    
    # Aggregate metrics
    results = {
        'overall_j_mae': float(np.mean(all_metrics['j_mae'])),
        'overall_j_rmse': float(np.mean(all_metrics['j_rmse'])),
        'overall_type_accuracy': float(np.mean(all_metrics['type_accuracy'])),
        'total_matched_pairs': int(sum(all_metrics['matched_pairs'])),
        'total_targets': int(sum(all_metrics['total_targets'])),
        'overall_matching_rate': float(np.mean(all_metrics['matching_rate'])),
        'median_j_mae': float(np.median(all_metrics['j_mae'])),
        'std_j_mae': float(np.std(all_metrics['j_mae']))
    }
    
    return results, predictions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='pseudo_labeled_dataset.csv',
                       help='Path to pseudo-labeled dataset')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate
    results, predictions = evaluate_model(
        args.checkpoint, 
        args.data,
        device=device
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall J-MAE: {results['overall_j_mae']:.4f} Hz")
    print(f"Overall J-RMSE: {results['overall_j_rmse']:.4f} Hz")
    print(f"Median J-MAE: {results['median_j_mae']:.4f} Hz")
    print(f"Std J-MAE: {results['std_j_mae']:.4f} Hz")
    print(f"Type Accuracy: {results['overall_type_accuracy']:.2%}")
    print(f"Matched Pairs: {results['total_matched_pairs']}/{results['total_targets']}")
    print(f"Matching Rate: {results['overall_matching_rate']:.2%}")
    print("="*60)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({'metrics': results, 'predictions': predictions}, f, indent=2)
    
    print(f"\n💾 Results saved to {args.output}")
    
    # Create DataFrame of predictions
    df_pred = pd.DataFrame(predictions)
    csv_output = args.output.replace('.json', '.csv')
    df_pred.to_csv(csv_output, index=False)
    print(f"💾 Predictions saved to {csv_output}")


if __name__ == '__main__':
    main()
