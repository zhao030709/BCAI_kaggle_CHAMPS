"""
Parse experimental NMR data from h_nmr column in filtered_test_dataset.csv.
Extract J-coupling constants (J_exp set) for matching with predictions.
"""
import pandas as pd
import numpy as np
import ast
import re
from typing import List, Dict, Tuple


def parse_h_nmr_string(h_nmr_str: str) -> List[Dict]:
    """
    Parse h_nmr column string to extract J-coupling constants.
    
    Format: [shift, integral, multiplicity, J_value, ...]
    Example: "[7.35, 10, 'm', 'nil', 3.51, 2, 't', 7.5, ...]"
    
    Returns:
        List of J-values (float), excluding 'nil' entries
    """
    try:
        # Parse string as list
        data = ast.literal_eval(h_nmr_str)
        
        j_values = []
        peak_info = []
        
        # Parse in chunks of 4: [shift, integral, multiplicity, J]
        i = 0
        peak_idx = 0
        while i < len(data):
            if i + 3 < len(data):
                shift = data[i]
                integral = data[i + 1]
                multiplicity = data[i + 2]
                j_val = data[i + 3]
                
                # Check if J value is numeric
                if j_val != 'nil' and isinstance(j_val, (int, float)):
                    j_values.append(float(j_val))
                    peak_info.append({
                        'peak_index': peak_idx,
                        'shift': shift,
                        'integral': integral,
                        'multiplicity': multiplicity,
                        'j_value': float(j_val)
                    })
                
                peak_idx += 1
                i += 4
            else:
                break
        
        return j_values, peak_info
    
    except Exception as e:
        print(f"Error parsing h_nmr: {e}")
        return [], []


def extract_j_values_from_dataset(csv_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Extract all J-coupling values from the dataset.
    
    Args:
        csv_file: Path to filtered_test_dataset.csv
        output_file: Optional output path for extracted data
    
    Returns:
        DataFrame with columns: [smiles, j_values, peak_info, n_j_values]
    """
    print(f"Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Total molecules: {len(df)}")
    print("Parsing h_nmr data...")
    
    results = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        h_nmr = row['h_nmr']
        
        j_values, peak_info = parse_h_nmr_string(h_nmr)
        
        results.append({
            'molecule_index': idx,
            'smiles': smiles,
            'j_values': j_values,
            'peak_info': peak_info,
            'n_j_values': len(j_values)
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} molecules...")
    
    result_df = pd.DataFrame(results)
    
    print("\nExtraction complete!")
    print(f"Molecules with J-values: {(result_df['n_j_values'] > 0).sum()}")
    print(f"Total J-values extracted: {result_df['n_j_values'].sum()}")
    print(f"Average J-values per molecule: {result_df['n_j_values'].mean():.2f}")
    
    # Statistics
    all_j_values = [j for jlist in result_df['j_values'] if len(jlist) > 0 for j in jlist]
    if len(all_j_values) > 0:
        print(f"\nJ-value statistics:")
        print(f"  Min: {min(all_j_values):.2f} Hz")
        print(f"  Max: {max(all_j_values):.2f} Hz")
        print(f"  Mean: {np.mean(all_j_values):.2f} Hz")
        print(f"  Median: {np.median(all_j_values):.2f} Hz")
    
    if output_file:
        result_df.to_pickle(output_file)
        print(f"\nSaved results to {output_file}")
    
    return result_df


def get_j_values_for_smiles(smiles: str, df: pd.DataFrame) -> Tuple[List[float], List[Dict]]:
    """
    Get J-values for a specific SMILES string.
    
    Args:
        smiles: SMILES string
        df: DataFrame from extract_j_values_from_dataset
    
    Returns:
        (j_values, peak_info) tuple
    """
    match = df[df['smiles'] == smiles]
    if len(match) == 0:
        return [], []
    
    row = match.iloc[0]
    return row['j_values'], row['peak_info']


def test_parser():
    """Test the parser on sample data."""
    test_cases = [
        "[10.05, 1, 's', 'nil', 7.54, 1, 'd', 2.7, 7.43, 1, 'd', 8.4]",
        "[7.35, 10, 'm', 'nil', 3.51, 2, 't', 7.5, 2.87, 2, 't', 7.5]",
        "[4.66, 2, 'q', 7.1, 3.75, 2, 's', 'nil', 3.5, 2, 's', 'nil']"
    ]
    
    print("Testing parser...")
    for i, test_str in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input: {test_str}")
        j_values, peak_info = parse_h_nmr_string(test_str)
        print(f"Extracted J-values: {j_values}")
        print(f"Peak info: {peak_info}")


if __name__ == '__main__':
    import sys
    import os
    
    # Test mode
    if len(sys.argv) == 1 or sys.argv[1] == 'test':
        test_parser()
    else:
        # Extract from full dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        csv_file = os.path.join(project_root, 'filtered_test_dataset.csv')
        output_file = os.path.join(project_root, 'experimental_j_values.pkl')
        
        df = extract_j_values_from_dataset(csv_file, output_file)
        
        # Show sample
        print("\nSample entries:")
        print(df[df['n_j_values'] > 0].head(3)[['smiles', 'j_values', 'n_j_values']])
