"""
使用训练好的Set Transformer模型对测试集进行预测。

输入: filtered_test_dataset_1000.csv (1000个测试分子)
输出: test_predictions_1000.csv (Kaggle提交格式)
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from set_transformer_model import create_model
from set_transformer_data import MolecularGraphDataset, collate_fn
from torch.utils.data import DataLoader
import argparse
import os


def predict_test_set(
    model_path,
    test_csv,
    output_csv,
    batch_size=8,
    device='cpu'
):
    """
    对测试集进行预测。
    
    Args:
        model_path: 训练好的模型checkpoint路径
        test_csv: 测试集CSV文件 (包含smiles列)
        output_csv: 输出预测结果CSV
        batch_size: 批大小
        device: 设备 (cpu/cuda)
    """
    print("="*60)
    print("Set Transformer Test Set Prediction")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = create_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # Load test data
    print(f"\nLoading test data from {test_csv}...")
    df_test = pd.read_csv(test_csv)
    
    # 检查是否有molecule_name列
    if 'molecule_name' not in df_test.columns:
        print("Warning: No molecule_name column, using index as molecule_name")
        df_test['molecule_name'] = [f'mol_{i}' for i in range(len(df_test))]
    
    smiles_list = df_test['smiles'].tolist()
    molecule_names = df_test['molecule_name'].tolist()
    
    print(f"Test set size: {len(smiles_list)} molecules")
    
    # Create dataset (without target couplings for test set)
    # 使用空列表作为占位符
    empty_couplings = [[] for _ in range(len(smiles_list))]
    
    dataset = MolecularGraphDataset(
        smiles_list,
        empty_couplings,
        max_atoms=config['model']['max_atoms']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    # Predict
    print("\nPredicting...")
    all_predictions = []
    
    # 类型编码映射 - 必须与训练时的MolecularGraphDataset一致！
    type_names = ['1JHC', '2JHC', '3JHC', '1JHH', '2JHH', '3JHH', '2JHN', '3JHN']
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            
            # Get predictions
            pred_j = output['j_values']  # [batch, max_atoms^2]
            pred_type_logits = output['type_logits']  # [batch, max_atoms^2, n_types]
            pred_types = torch.argmax(pred_type_logits, dim=-1)  # [batch, max_atoms^2]
            pred_mask = batch['pred_mask']  # [batch, max_atoms^2]
            
            # Extract atom types
            atom_types_batch = batch['atom_types']  # [batch, max_atoms]
            
            # Process each molecule in batch
            batch_size_actual = len(pred_j)
            for i in range(batch_size_actual):
                mol_idx = batch_idx * batch_size + i
                if mol_idx >= len(molecule_names):
                    break
                
                mol_name = molecule_names[mol_idx]
                smiles = smiles_list[mol_idx]
                
                # Get valid atoms (non-padding)
                atom_types = atom_types_batch[i].cpu().numpy()
                valid_atoms = atom_types > 0
                n_atoms = valid_atoms.sum()
                
                # Get valid predictions
                valid_pred_idx = pred_mask[i].cpu().numpy()
                
                # Iterate through all atom pairs
                max_atoms = config['model']['max_atoms']
                for pair_flat_idx in range(len(valid_pred_idx)):
                    if not valid_pred_idx[pair_flat_idx]:
                        continue
                    
                    # Convert flat index to (atom_i, atom_j)
                    # 模型输出是max_atoms x max_atoms的展平
                    atom_i = pair_flat_idx // max_atoms
                    atom_j = pair_flat_idx % max_atoms
                    
                    # 跳过padding原子
                    if atom_i >= n_atoms or atom_j >= n_atoms:
                        continue
                    
                    # 跳过自环
                    if atom_i == atom_j:
                        continue
                    
                    j_value = float(pred_j[i, pair_flat_idx].cpu().item())
                    coupling_type = int(pred_types[i, pair_flat_idx].cpu().item())
                    
                    # 过滤异常值
                    if j_value < 0 or j_value > 300:  # J值通常在0-300 Hz范围
                        continue
                    
                    type_str = type_names[coupling_type] if coupling_type < len(type_names) else 'unknown'
                    
                    all_predictions.append({
                        'molecule_name': mol_name,
                        'atom_index_0': int(atom_i),
                        'atom_index_1': int(atom_j),
                        'scalar_coupling_constant': j_value,
                        'type': type_str
                    })
    
    # Create output DataFrame
    df_pred = pd.DataFrame(all_predictions)
    
    print(f"\n✅ Generated {len(df_pred)} predictions")
    print(f"   From {len(set(df_pred['molecule_name']))} molecules")
    
    # Statistics
    print("\nPrediction statistics:")
    print(f"  J-coupling range: [{df_pred['scalar_coupling_constant'].min():.2f}, {df_pred['scalar_coupling_constant'].max():.2f}] Hz")
    print(f"  J-coupling mean: {df_pred['scalar_coupling_constant'].mean():.2f} Hz")
    print(f"  J-coupling median: {df_pred['scalar_coupling_constant'].median():.2f} Hz")
    
    print("\nType distribution:")
    type_counts = df_pred['type'].value_counts()
    for t, count in type_counts.items():
        print(f"  {t}: {count} ({count/len(df_pred)*100:.1f}%)")
    
    # Save
    df_pred.to_csv(output_csv, index=False)
    print(f"\n💾 Predictions saved to {output_csv}")
    
    return df_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test', type=str, default='filtered_test_dataset_1000.csv',
                       help='Test dataset CSV file')
    parser.add_argument('--output', type=str, default='test_predictions_1000.csv',
                       help='Output predictions CSV file')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for prediction')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    predictions = predict_test_set(
        args.checkpoint,
        args.test,
        args.output,
        batch_size=args.batch_size,
        device=device
    )
    
    print("\n" + "="*60)
    print("Prediction completed!")
    print("="*60)


if __name__ == '__main__':
    main()
