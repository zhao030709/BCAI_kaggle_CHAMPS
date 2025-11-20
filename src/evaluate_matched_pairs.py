"""
评估matched_pairs文件中的预测质量
基于已有的matched_pairs_1000_original.csv进行详细分析
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

def evaluate_matched_pairs(matched_pairs_file):
    """
    评估matched pairs的质量
    """
    print(f"Loading matched pairs from {matched_pairs_file}...")
    df = pd.read_csv(matched_pairs_file)
    print(f"  Loaded {len(df)} matched pairs")
    
    # 计算误差统计
    errors = np.abs(df['error'])
    
    results = {
        'total_pairs': len(df),
        'overall_metrics': {
            'j_mae': float(np.mean(errors)),
            'j_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'percentile_25': float(np.percentile(errors, 25)),
            'percentile_75': float(np.percentile(errors, 75)),
            'percentile_90': float(np.percentile(errors, 90)),
            'percentile_95': float(np.percentile(errors, 95)),
        }
    }
    
    # 误差分布
    error_bins = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, float('inf')]
    error_labels = ['0-0.05 Hz', '0.05-0.1 Hz', '0.1-0.2 Hz', '0.2-0.5 Hz', '0.5-1.0 Hz', '>1.0 Hz']
    
    error_distribution = {}
    for i in range(len(error_bins) - 1):
        count = np.sum((errors >= error_bins[i]) & (errors < error_bins[i+1]))
        percentage = count / len(errors) * 100
        error_distribution[error_labels[i]] = {
            'count': int(count),
            'percentage': float(percentage)
        }
    
    results['error_distribution'] = error_distribution
    
    # 按类型统计
    by_type = {}
    for coupling_type in df['type'].unique():
        type_df = df[df['type'] == coupling_type]
        type_errors = np.abs(type_df['error'])
        
        by_type[coupling_type] = {
            'count': len(type_df),
            'j_mae': float(np.mean(type_errors)),
            'j_rmse': float(np.sqrt(np.mean(type_errors ** 2))),
            'median_error': float(np.median(type_errors)),
            'j_pred_mean': float(type_df['j_pred'].mean()),
            'j_exp_mean': float(type_df['j_exp'].mean()),
            'j_pred_std': float(type_df['j_pred'].std()),
            'j_exp_std': float(type_df['j_exp'].std()),
        }
    
    results['by_type'] = by_type
    
    # 按分子统计
    mol_errors = df.groupby('smiles')['error'].apply(lambda x: np.abs(x).mean()).values
    
    results['per_molecule'] = {
        'unique_molecules': int(df['smiles'].nunique()),
        'avg_pairs_per_molecule': float(len(df) / df['smiles'].nunique()),
        'avg_error_per_molecule': float(np.mean(mol_errors)),
        'median_error_per_molecule': float(np.median(mol_errors)),
    }
    
    return results, df


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("MATCHED PAIRS EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n总体统计:")
    print(f"  配对总数: {results['total_pairs']}")
    print(f"  唯一分子数: {results['per_molecule']['unique_molecules']}")
    print(f"  平均每分子配对数: {results['per_molecule']['avg_pairs_per_molecule']:.1f}")
    
    print(f"\n整体性能:")
    metrics = results['overall_metrics']
    print(f"  J-MAE: {metrics['j_mae']:.4f} Hz ⭐")
    print(f"  J-RMSE: {metrics['j_rmse']:.4f} Hz")
    print(f"  中位误差: {metrics['median_error']:.4f} Hz")
    print(f"  标准差: {metrics['std_error']:.4f} Hz")
    print(f"  误差范围: [{metrics['min_error']:.4f}, {metrics['max_error']:.4f}] Hz")
    
    print(f"\n误差分布:")
    for label, data in results['error_distribution'].items():
        print(f"  {label:12s}: {data['percentage']:5.1f}% ({data['count']:4d} pairs)")
    
    print(f"\n分位数:")
    print(f"  25th percentile: {metrics['percentile_25']:.4f} Hz")
    print(f"  75th percentile: {metrics['percentile_75']:.4f} Hz")
    print(f"  90th percentile: {metrics['percentile_90']:.4f} Hz")
    print(f"  95th percentile: {metrics['percentile_95']:.4f} Hz")
    
    print(f"\n按类型统计:")
    for coupling_type, data in sorted(results['by_type'].items()):
        print(f"  {coupling_type}:")
        print(f"    Count: {data['count']}")
        print(f"    J-MAE: {data['j_mae']:.4f} Hz")
        print(f"    预测均值: {data['j_pred_mean']:.4f} Hz (实验: {data['j_exp_mean']:.4f} Hz)")
        print(f"    差异: {data['j_pred_mean'] - data['j_exp_mean']:+.4f} Hz")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='评估matched pairs文件')
    parser.add_argument('--input', type=str, default='archived/matched_pairs_1000_original.csv',
                        help='Matched pairs CSV文件路径')
    parser.add_argument('--output', type=str, default='outputs/evaluation/matched_pairs_evaluation.json',
                        help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    results, df = evaluate_matched_pairs(args.input)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved evaluation to {output_path}")
    
    # 打印结果
    print_results(results)


if __name__ == '__main__':
    main()
