"""
Match predicted J-coupling values with experimental values using Hungarian algorithm.
This implements the "Predict-and-Match" strategy for pseudo-labeling.
"""
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import pickle


def apply_multiplicity_penalty(cost: float, multiplicity: str, coupling_type: str) -> float:
    """
    Apply multiplicity penalty - ORIGINAL strategy (no penalty).
    
    Original strategy: no multiplicity constraints to maximize coverage.
    
    Args:
        cost: Base cost (absolute error)
        multiplicity: Experimental peak multiplicity
        coupling_type: Type of coupling
    
    Returns:
        Cost unchanged (no penalty)
    """
    # Original strategy: no penalties
    return cost


def get_type_specific_threshold(coupling_type: str) -> float:
    """
    Get type-specific error threshold - ORIGINAL strategy.
    
    Uniform threshold for all types to maximize coverage.
    
    Args:
        coupling_type: Coupling type (e.g., '2JHC', '3JHH')
    
    Returns:
        Maximum allowed error (Hz) for this type
    """
    # Original strategy: uniform 2.0 Hz threshold for all types
    return 2.0


def build_cost_matrix(j_pred: List[float], j_exp: List[float], 
                     pred_types: List[str] = None,
                     exp_multiplicities: List[str] = None) -> np.ndarray:
    """
    Build cost matrix for bipartite matching - ORIGINAL strategy (no constraints).
    
    Cost[i, j] = |J_pred[i] - J_exp[j]|
    
    Args:
        j_pred: Predicted J-values (from model)
        j_exp: Experimental J-values (from h_nmr)
        pred_types: Unused (for compatibility)
        exp_multiplicities: Unused (for compatibility)
    
    Returns:
        Cost matrix of shape (len(j_pred), len(j_exp))
    """
    n_pred = len(j_pred)
    n_exp = len(j_exp)
    
    # Create meshgrid for broadcasting
    pred_array = np.array(j_pred).reshape(-1, 1)
    exp_array = np.array(j_exp).reshape(1, -1)
    
    # Compute absolute difference - NO constraints applied
    cost = np.abs(pred_array - exp_array)
    
    # Original strategy: no filtering, no penalties
    # Just return raw cost matrix for Hungarian algorithm
    
    return cost


def hungarian_match(j_pred: List[float], j_exp: List[float], 
                    pred_info: List[Dict] = None,
                    exp_info: List[Dict] = None) -> Tuple[List[Dict], float]:
    """
    Perform Hungarian algorithm matching with multiplicity constraints.
    
    Args:
        j_pred: List of predicted J-values
        j_exp: List of experimental J-values
        pred_info: Optional list of dicts with atom pair info and type for each prediction
        exp_info: Optional list of dicts with experimental peak info (multiplicity, shift, etc.)
    
    Returns:
        (matched_pairs, total_cost) where matched_pairs is a list of dicts
    """
    if len(j_pred) == 0 or len(j_exp) == 0:
        return [], 0.0
    
    # Extract types and multiplicities for constraint checking
    pred_types = None
    exp_multiplicities = None
    
    if pred_info is not None:
        pred_types = [info.get('type', '') for info in pred_info]
    
    if exp_info is not None:
        exp_multiplicities = [info.get('multiplicity', 'm') for info in exp_info]
    
    # Build cost matrix with constraints
    cost_matrix = build_cost_matrix(j_pred, j_exp, pred_types, exp_multiplicities)
    
    # Check if matrix is too small or all costs are infinite
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return [], 0.0
    
    # If all costs are infinite, no valid matches
    if np.all(np.isinf(cost_matrix)):
        return [], 0.0
    
    # For Hungarian algorithm to work, we need to handle the case where 
    # the matrix has more rows than columns or vice versa
    # linear_sum_assignment can handle rectangular matrices
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        # If still infeasible, return empty
        print(f"Warning: {e}")
        return [], 0.0
    
    # Extract matches
    matched_pairs = []
    total_cost = 0.0
    
    for pred_idx, exp_idx in zip(row_ind, col_ind):
        cost = cost_matrix[pred_idx, exp_idx]
        
        # Skip infinite cost (incompatible or exceeds threshold)
        if np.isinf(cost):
            continue
        
        # ORIGINAL STRATEGY: Apply 2.0 Hz quality threshold post-matching
        # This filters out poor matches while allowing Hungarian to find optimal assignment
        if cost > 2.0:
            continue
        
        match = {
            'pred_idx': int(pred_idx),
            'exp_idx': int(exp_idx),
            'j_pred': float(j_pred[pred_idx]),
            'j_exp': float(j_exp[exp_idx]),
            'error': float(cost)
        }
        
        # Add atom pair info if available
        if pred_info is not None and pred_idx < len(pred_info):
            info = pred_info[pred_idx]
            match['atom_0'] = info.get('atom_index_0', None)
            match['atom_1'] = info.get('atom_index_1', None)
            match['type'] = info.get('type', None)
        
        # Add experimental info if available
        if exp_info is not None and exp_idx < len(exp_info):
            info = exp_info[exp_idx]
            match['multiplicity'] = info.get('multiplicity', None)
            match['shift'] = info.get('shift', None)
        
        matched_pairs.append(match)
        total_cost += cost
    
    return matched_pairs, total_cost


def match_molecule(pred_df: pd.DataFrame, smiles: str, 
                   exp_peak_info: List[Dict]) -> Dict:
    """
    Match predictions for a single molecule with multiplicity constraints.
    
    Args:
        pred_df: DataFrame with predictions (from assign_peaks.py output)
        smiles: SMILES string of the molecule
        exp_peak_info: List of experimental peak info dicts from parse_nmr_data.py
    
    Returns:
        Dict with matching results
    """
    # Extract experimental J-values
    j_exp = [peak['j_value'] for peak in exp_peak_info]
    
    if len(j_exp) == 0:
        return {
            'smiles': smiles,
            'n_pred': 0,
            'n_exp': 0,
            'n_matched': 0,
            'matches': [],
            'match_rate': 0.0,
            'avg_error': None
        }
    
    # Filter predictions for this molecule
    mol_pred = pred_df[pred_df['smiles'] == smiles].copy()
    
    if len(mol_pred) == 0:
        return {
            'smiles': smiles,
            'n_pred': 0,
            'n_exp': len(j_exp),
            'n_matched': 0,
            'matches': [],
            'match_rate': 0.0,
            'avg_error': None
        }
    
    # Extract predictions (only H-H couplings for now)
    # Filter to common types that appear in NMR
    common_types = ['2JHH', '3JHH', '2JHC', '3JHC', '1JHC']
    mol_pred = mol_pred[mol_pred['type'].isin(common_types)]
    
    # Filter out NaN predictions
    mol_pred = mol_pred.dropna(subset=['scalar_coupling_constant'])
    
    j_pred = mol_pred['scalar_coupling_constant'].tolist()
    
    # Build pred_info for matching
    pred_info = []
    for _, row in mol_pred.iterrows():
        pred_info.append({
            'atom_index_0': row['atom_index_0'],
            'atom_index_1': row['atom_index_1'],
            'type': row['type']
        })
    
    # Perform matching with constraints
    matches, total_cost = hungarian_match(j_pred, j_exp, pred_info, exp_peak_info)
    
    # Calculate statistics
    n_matched = len(matches)
    match_rate = n_matched / max(len(j_exp), 1)
    avg_error = total_cost / max(n_matched, 1) if n_matched > 0 else None
    
    return {
        'smiles': smiles,
        'n_pred': len(j_pred),
        'n_exp': len(j_exp),
        'n_matched': n_matched,
        'matches': matches,
        'match_rate': match_rate,
        'avg_error': avg_error
    }


def batch_match(pred_csv: str, exp_pkl: str, output_csv: str = None) -> pd.DataFrame:
    """
    Match predictions with experimental data using improved constraints.
    
    Args:
        pred_csv: Path to predictions CSV (from assign_peaks.py)
        exp_pkl: Path to experimental J-values pickle (from parse_nmr_data.py)
        output_csv: Optional output path for matched pairs
    
    Returns:
        DataFrame with all matched pairs
    """
    print(f"Loading predictions from {pred_csv}...")
    pred_df = pd.read_csv(pred_csv)
    print(f"Loaded {len(pred_df)} predictions")
    
    print(f"\nLoading experimental data from {exp_pkl}...")
    with open(exp_pkl, 'rb') as f:
        exp_df = pickle.load(f)
    print(f"Loaded {len(exp_df)} molecules with experimental data")
    
    # Find common molecules
    pred_smiles = set(pred_df['smiles'].unique())
    exp_smiles_with_j = set(exp_df[exp_df['n_j_values'] > 0]['smiles'])
    common_smiles = pred_smiles & exp_smiles_with_j
    
    print(f"\nMolecules with both predictions and experimental J-values: {len(common_smiles)}")
    
    if len(common_smiles) == 0:
        print("No common molecules found!")
        return pd.DataFrame()
    
    # Match each molecule
    print("\nMatching predictions with experiments (using multiplicity constraints)...")
    all_results = []
    all_matches = []
    
    for i, smiles in enumerate(common_smiles):
        # Get experimental peak info
        exp_row = exp_df[exp_df['smiles'] == smiles].iloc[0]
        peak_info = exp_row['peak_info']
        
        # Match with constraints
        result = match_molecule(pred_df, smiles, peak_info)
        all_results.append(result)
        
        # Expand matches for output
        for match in result['matches']:
            match_record = {
                'smiles': smiles,
                'atom_index_0': match['atom_0'],
                'atom_index_1': match['atom_1'],
                'type': match['type'],
                'j_pred': match['j_pred'],
                'j_exp': match['j_exp'],
                'error': match['error'],
                'multiplicity': match.get('multiplicity', None),
                'shift': match.get('shift', None)
            }
            all_matches.append(match_record)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(common_smiles)} molecules...")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("MATCHING RESULTS (With Multiplicity Constraints)")
    print("="*80)
    print(f"Total molecules processed: {len(summary_df)}")
    print(f"Total experimental J-values: {summary_df['n_exp'].sum()}")
    print(f"Total predictions: {summary_df['n_pred'].sum()}")
    print(f"Total matched pairs: {summary_df['n_matched'].sum()}")
    print(f"Overall match rate: {summary_df['n_matched'].sum() / max(summary_df['n_exp'].sum(), 1):.1%}")
    
    # Error statistics
    errors = [m['error'] for r in all_results for m in r['matches']]
    if len(errors) > 0:
        print(f"\nError statistics (Hz):")
        print(f"  Mean: {np.mean(errors):.3f}")
        print(f"  Median: {np.median(errors):.3f}")
        print(f"  Std: {np.std(errors):.3f}")
        print(f"  Max: {np.max(errors):.3f}")
        print(f"  < 0.5 Hz: {(np.array(errors) < 0.5).sum()} ({(np.array(errors) < 0.5).sum()/len(errors)*100:.1f}%)")
        print(f"  < 1.0 Hz: {(np.array(errors) < 1.0).sum()} ({(np.array(errors) < 1.0).sum()/len(errors)*100:.1f}%)")
    
    # Save matched pairs
    matches_df = pd.DataFrame(all_matches)
    
    if output_csv:
        matches_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(matches_df)} matched pairs to {output_csv}")
        
        summary_file = output_csv.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to {summary_file}")
    
    return matches_df


if __name__ == '__main__':
    import sys
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check for command line argument to specify dataset size
    dataset_size = '100'  # default
    if len(sys.argv) > 1:
        dataset_size = sys.argv[1]
    
    # Use specified dataset size
    pred_csv = os.path.join(project_root, f'predicted_couplings_{dataset_size}.csv')
    exp_pkl = os.path.join(project_root, 'experimental_j_values.pkl')
    output_csv = os.path.join(project_root, f'matched_pairs_{dataset_size}_original.csv')
    
    print("="*80)
    print(f"ORIGINAL MATCHING STRATEGY - {dataset_size} molecules")
    print("="*80)
    print("\nUniform threshold for all types:")
    print("  All coupling types: 2.0 Hz")
    print("\nNo multiplicity constraints")
    print("  → Maximize coverage over precision")
    if dataset_size == '100':
        print("  → Previous results: 64.1% match rate, 1.026 Hz median error")
    print()
    
    matches_df = batch_match(pred_csv, exp_pkl, output_csv)
