"""
分析测试集预测的统计特性
与训练集的伪标签进行对比
"""

import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path

def analyze_predictions(predictions_df, ground_truth_df):
    """
    分析预测结果的统计特性
    """
    results = {}
    
    # 基本统计
    results['predictions'] = {
        'total_count': len(predictions_df),
        'unique_molecules': predictions_df['molecule_name'].nunique() if 'molecule_name' in predictions_df.columns else 'N/A',
        'j_value_stats': {
            'mean': float(predictions_df['scalar_coupling_constant'].mean()),
            'median': float(predictions_df['scalar_coupling_constant'].median()),
            'std': float(predictions_df['scalar_coupling_constant'].std()),
            'min': float(predictions_df['scalar_coupling_constant'].min()),
            'max': float(predictions_df['scalar_coupling_constant'].max()),
            'percentile_25': float(predictions_df['scalar_coupling_constant'].quantile(0.25)),
            'percentile_75': float(predictions_df['scalar_coupling_constant'].quantile(0.75)),
        }
    }
    
    # 类型分布
    type_dist_pred = predictions_df['type'].value_counts()
    type_pct_pred = (type_dist_pred / len(predictions_df) * 100).to_dict()
    results['predictions']['type_distribution'] = {
        'counts': type_dist_pred.to_dict(),
        'percentages': type_pct_pred
    }
    
    # Ground truth统计
    j_col_gt = 'j_value' if 'j_value' in ground_truth_df.columns else 'scalar_coupling_constant'
    results['ground_truth'] = {
        'total_count': len(ground_truth_df),
        'j_value_stats': {
            'mean': float(ground_truth_df[j_col_gt].mean()),
            'median': float(ground_truth_df[j_col_gt].median()),
            'std': float(ground_truth_df[j_col_gt].std()),
            'min': float(ground_truth_df[j_col_gt].min()),
            'max': float(ground_truth_df[j_col_gt].max()),
            'percentile_25': float(ground_truth_df[j_col_gt].quantile(0.25)),
            'percentile_75': float(ground_truth_df[j_col_gt].quantile(0.75)),
        }
    }
    
    # 类型分布
    type_dist_gt = ground_truth_df['type'].value_counts()
    type_pct_gt = (type_dist_gt / len(ground_truth_df) * 100).to_dict()
    results['ground_truth']['type_distribution'] = {
        'counts': type_dist_gt.to_dict(),
        'percentages': type_pct_gt
    }
    
    # 按类型分析J值分布
    results['by_type_comparison'] = {}
    all_types = set(predictions_df['type'].unique()) | set(ground_truth_df['type'].unique())
    
    for coupling_type in sorted(all_types):
        pred_type = predictions_df[predictions_df['type'] == coupling_type]
        gt_type = ground_truth_df[ground_truth_df['type'] == coupling_type]
        
        results['by_type_comparison'][coupling_type] = {
            'predictions': {
                'count': len(pred_type),
                'j_mean': float(pred_type['scalar_coupling_constant'].mean()) if len(pred_type) > 0 else None,
                'j_std': float(pred_type['scalar_coupling_constant'].std()) if len(pred_type) > 0 else None,
            },
            'ground_truth': {
                'count': len(gt_type),
                'j_mean': float(gt_type[j_col_gt].mean()) if len(gt_type) > 0 else None,
                'j_std': float(gt_type[j_col_gt].std()) if len(gt_type) > 0 else None,
            }
        }
        
        if len(pred_type) > 0 and len(gt_type) > 0:
            results['by_type_comparison'][coupling_type]['mean_difference'] = \
                results['by_type_comparison'][coupling_type]['predictions']['j_mean'] - \
                results['by_type_comparison'][coupling_type]['ground_truth']['j_mean']
    
    # 分布相似性分析
    results['distribution_similarity'] = {
        'j_value_mean_diff': results['predictions']['j_value_stats']['mean'] - 
                             results['ground_truth']['j_value_stats']['mean'],
        'j_value_median_diff': results['predictions']['j_value_stats']['median'] - 
                               results['ground_truth']['j_value_stats']['median'],
        'type_distribution_correlation': None  # 可以计算相关系数
    }
    
    return results


def print_results(results):
    """打印分析结果"""
    print("\n" + "="*60)
    print("PREDICTION STATISTICS ANALYSIS")
    print("="*60)
    
    print(f"\n预测结果统计:")
    pred = results['predictions']
    print(f"  总预测数: {pred['total_count']}")
    print(f"  分子数: {pred['unique_molecules']}")
    print(f"  J值统计:")
    print(f"    均值: {pred['j_value_stats']['mean']:.4f} Hz")
    print(f"    中位数: {pred['j_value_stats']['median']:.4f} Hz")
    print(f"    标准差: {pred['j_value_stats']['std']:.4f} Hz")
    print(f"    范围: [{pred['j_value_stats']['min']:.4f}, {pred['j_value_stats']['max']:.4f}] Hz")
    
    print(f"\n  类型分布:")
    for ctype, pct in sorted(pred['type_distribution']['percentages'].items(), 
                             key=lambda x: x[1], reverse=True):
        count = pred['type_distribution']['counts'][ctype]
        print(f"    {ctype}: {pct:.1f}% ({count})")
    
    print(f"\n训练集Ground Truth统计:")
    gt = results['ground_truth']
    print(f"  总耦合数: {gt['total_count']}")
    print(f"  J值统计:")
    print(f"    均值: {gt['j_value_stats']['mean']:.4f} Hz")
    print(f"    中位数: {gt['j_value_stats']['median']:.4f} Hz")
    print(f"    标准差: {gt['j_value_stats']['std']:.4f} Hz")
    print(f"    范围: [{gt['j_value_stats']['min']:.4f}, {gt['j_value_stats']['max']:.4f}] Hz")
    
    print(f"\n  类型分布:")
    for ctype, pct in sorted(gt['type_distribution']['percentages'].items(), 
                             key=lambda x: x[1], reverse=True):
        count = gt['type_distribution']['counts'][ctype]
        print(f"    {ctype}: {pct:.1f}% ({count})")
    
    print(f"\n分布相似性:")
    sim = results['distribution_similarity']
    print(f"  J值均值差异: {sim['j_value_mean_diff']:+.4f} Hz")
    print(f"  J值中位数差异: {sim['j_value_median_diff']:+.4f} Hz")
    
    print(f"\n按类型对比:")
    for ctype, data in sorted(results['by_type_comparison'].items()):
        pred_data = data['predictions']
        gt_data = data['ground_truth']
        print(f"  {ctype}:")
        print(f"    预测: count={pred_data['count']}, J-mean={pred_data['j_mean']:.4f} Hz" if pred_data['j_mean'] else f"    预测: count=0")
        print(f"    训练: count={gt_data['count']}, J-mean={gt_data['j_mean']:.4f} Hz" if gt_data['j_mean'] else f"    训练: count=0")
        if 'mean_difference' in data:
            print(f"    差异: {data['mean_difference']:+.4f} Hz")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='分析测试集预测的统计特性')
    parser.add_argument('--predictions', type=str, required=True,
                        help='预测结果CSV文件路径')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='训练集伪标签CSV文件路径')
    parser.add_argument('--output', type=str, default='prediction_statistics.json',
                        help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}...")
    predictions_df = pd.read_csv(args.predictions)
    print(f"  Loaded {len(predictions_df)} predictions")
    
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth_df = pd.read_csv(args.ground_truth)
    print(f"  Loaded {len(ground_truth_df)} ground truth couplings")
    
    print("\nAnalyzing...")
    results = analyze_predictions(predictions_df, ground_truth_df)
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved analysis to {output_path}")
    
    # 打印结果
    print_results(results)


if __name__ == '__main__':
    main()
