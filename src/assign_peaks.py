import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
import gzip
import importlib
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import functools

# Ensure we are running from src to satisfy relative path assumptions in pipeline_pre
if os.path.basename(os.getcwd()) != 'src':
    print("Please run this script from the 'src' directory.")
    print("Example: cd src && python assign_peaks.py")
    sys.exit(1)

sys.path.append(os.getcwd())

# Import pipeline functions
try:
    from pipeline_pre import (
        make_structure_dict,
        enhance_structure_dict,
        enhance_bonds,
        add_all_pairs,
        make_triplets,
        make_quadruplets,
        add_embedding,
        add_scaling,
        create_dataset,
        atomic_num_dict,
        get_scaling,
        enhance_atoms
    )
except ImportError:
    print("Error: Could not import pipeline_pre. Make sure you are in the src directory.")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem

# ==========================================
# Predictor Class (Adapted from demo_predictor.py)
# ==========================================
class Predictor:
    def __init__(self, model_name, model_dir='../models', config_dir='../config', device='cuda'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.config_dir = config_dir
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        # Load models.json to get directory mapping
        with open(os.path.join(self.config_dir, 'models.json')) as f:
            models_config = json.load(f)
        
        model_subdir = models_config.get(self.model_name + '_dir')
        if not model_subdir:
            # Fallback: assume model_name IS the directory name if not found
            model_subdir = self.model_name
            
        model_folder = os.path.join(self.model_dir, model_subdir)
        if not os.path.isdir(model_folder):
            raise ValueError(f"Model folder not found: {model_folder}")
            
        ckpt_path = os.path.join(model_folder, 'model.ckpt')
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")

        # Dynamic import of graph_transformer from model folder
        sys.path.insert(0, model_folder)
        try:
            import graph_transformer
            importlib.reload(graph_transformer)
        finally:
            sys.path.pop(0)

        # Load config
        with open(os.path.join(model_folder, 'config')) as f:
            config_str = f.read().replace("'", '"')
            config = json.loads(config_str)

        # Clean config (legacy cleanup)
        to_del = ['name','optim','lr','mom','scheduler','warmup_step','decay_rate',
                'lr_min','clip','max_epoch','batch_size','seed','cuda','debug',
                'patience','champs_loss','multi_gpu','fp16','max_bond_count',
                'log_interval','batch_chunk','work_dir','restart','restart_dir',
                'load','mode','eta_min','gpu0_bsz','n_all_param','max_step','d_embed',
                'cutout']
        for new,old in {'dim':'d_model', 'n_layers':'n_layer', 'fdim':'feature_dim',
                'dist_embedding':'dist_embed_type', 'atom_angle_embedding':'angle_embed_type',
                'trip_angle_embedding':'quad_angle_embed_type',
                'quad_angle_embedding':'quad_angle_embed_type',
                }.items():
            if old in config and new not in config:
                config[new] = config[old]
                to_del.append(old)
        for old in to_del:
            if old in config:
                del config[old]
        
        # Add type counts from models.json
        config.update({k:models_config[k] for k in models_config if k.startswith('num') and k.endswith('types')})
        
        # Initialize model
        model = graph_transformer.GraphTransformer(**config)
        
        # Load weights
        map_location = 'cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(ckpt_path, map_location=map_location)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, batch_tensors):
        """
        Run inference on a batch of tensors.
        batch_tensors: tuple of tensors as returned by create_dataset (but batched)
        """
        MAX_BOND_COUNT = 406 # From predictor.py
        
        x_idx, x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle, y = batch_tensors
        
        # Move to device
        x_atom = x_atom.to(self.device)
        x_atom_pos = x_atom_pos.to(self.device)
        x_bond = x_bond.to(self.device)
        x_bond_dist = x_bond_dist.to(self.device)
        x_triplet = x_triplet.to(self.device)
        x_triplet_angle = x_triplet_angle.to(self.device)
        x_quad = x_quad.to(self.device)
        x_quad_angle = x_quad_angle.to(self.device)
        y = y.to(self.device)

        # Truncate to MAX_BOND_COUNT
        x_bond = x_bond[:, :MAX_BOND_COUNT]
        x_bond_dist = x_bond_dist[:, :MAX_BOND_COUNT]
        y = y[:, :MAX_BOND_COUNT]

        with torch.no_grad():
            y_pred, _ = self.model(x_atom, x_atom_pos, x_bond, x_bond_dist, x_triplet, x_triplet_angle, x_quad, x_quad_angle)
            
            # Post-processing (scaling back)
            y_pred_pad = torch.cat([torch.zeros(y_pred.shape[0], 1, y_pred.shape[2], device=y_pred.device), y_pred], dim=1)
            
            # Gather based on bond type
            # y[:,:,2] is std, y[:,:,1] is mean
            bond_types = x_bond[:,:,1].unsqueeze(-1) # [batch, bonds, 1]
            
            gathered = y_pred_pad.gather(1, bond_types) # [batch, bonds, 1]
            
            # Scale: pred * std + mean
            y_pred_scaled = gathered[:,:,0] * y[:,:,2] + y[:,:,1]
            
            return y_pred_scaled, x_bond, y

# ==========================================
# Helper Functions
# ==========================================

def build_embeddings_and_scaling(processed_dir='../processed'):
    """
    Load processed training data to reconstruct embeddings and scaling factors.
    This ensures we use the same mapping as the model training.
    """
    print("Loading processed data to build embeddings (this may take a minute)...")
    
    try:
        # Load structures
        structures_path = os.path.join(processed_dir, 'new_big_structures.csv.bz2')
        atoms = pd.read_csv(structures_path)
        
        # Load train bonds (we need types)
        train_path = os.path.join(processed_dir, 'new_big_train.csv.bz2')
        bonds = pd.read_csv(train_path)
        
        print(f"Loaded {len(bonds)} training bonds")
        print(f"Columns: {bonds.columns.tolist()}")
        
        # --- Critical Data Validation ---
        if 'scalar_coupling_constant' not in bonds.columns:
            raise ValueError("Missing 'scalar_coupling_constant' column in training data")
        
        # Check data type
        scc_dtype = bonds['scalar_coupling_constant'].dtype
        print(f"scalar_coupling_constant dtype: {scc_dtype}")
        
        # Force conversion to numeric if needed
        if not pd.api.types.is_numeric_dtype(bonds['scalar_coupling_constant']):
            print(f"WARNING: scalar_coupling_constant is {scc_dtype}, forcing numeric conversion...")
            original_count = len(bonds)
            
            # Show problematic values
            print(f"First 10 values before conversion: {bonds['scalar_coupling_constant'].head(10).tolist()}")
            
            # Convert to numeric, coerce errors to NaN
            bonds['scalar_coupling_constant'] = pd.to_numeric(bonds['scalar_coupling_constant'], errors='coerce')
            
            # Count conversion failures
            nan_count = bonds['scalar_coupling_constant'].isna().sum()
            if nan_count > 0:
                print(f"ERROR: {nan_count}/{original_count} values could not be converted to numeric!")
                print("Sample of failed rows:")
                print(bonds[bonds['scalar_coupling_constant'].isna()].head(10))
                raise ValueError("Data corruption detected in scalar_coupling_constant column")
            
            print("✓ Successfully converted all values to numeric")
        else:
            print(f"✓ scalar_coupling_constant is already numeric ({scc_dtype})")
            print(f"  Range: [{bonds['scalar_coupling_constant'].min():.2f}, {bonds['scalar_coupling_constant'].max():.2f}]")
            print(f"  Mean: {bonds['scalar_coupling_constant'].mean():.2f}")
        
        # ----------------------

        # Load triplets/quads if needed for types
        triplets_path = os.path.join(processed_dir, 'new_big_train_triplets.csv.bz2')
        triplets = pd.read_csv(triplets_path)
        
        quads_path = os.path.join(processed_dir, 'new_big_train_quadruplets.csv.bz2')
        quads = pd.read_csv(quads_path)
        
        print("Reconstructing embeddings...")
        # We pass embeddings=None to generate them
        embeddings = add_embedding(atoms, bonds, triplets, quads, embeddings=None)
        
        print("Calculating scaling factors...")
        means, stds = get_scaling(bonds)
        
        print(f"✓ Successfully computed scaling for {len(means)} bond types")
        
        return embeddings, means, stds
        
    except Exception as e:
        print(f"Error building embeddings: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure you have run the preprocessing pipeline (pipeline_pre.py) first.")
        sys.exit(1)

def process_single_molecule(row_tuple, embeddings, means, stds, cache_dir=None):
    """
    Worker function to process a single molecule.
    返回值简化：只返回成功/失败状态，结果直接写入缓存
    """
    idx, row = row_tuple
    
    # Check cache first
    if cache_dir:
        cache_path = os.path.join(cache_dir, f'mol_{idx}.pkl.gz')
        if os.path.exists(cache_path):
            return {'index': idx, 'status': 'cached'}

    smiles = row['smiles']
    mol_name = f'mol_{idx}'
    
    try:
        # 1. Generate 3D Conformer (添加超时保护)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'index': idx, 'status': 'invalid_smiles'}
        
        mol = Chem.AddHs(mol)
        
        # 限制尝试次数，防止卡死
        res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=5)
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42, maxAttempts=3)
        
        if res != 0:
            return {'index': idx, 'status': 'embed_failed'}
        
        # 限制优化迭代次数
        AllChem.MMFFOptimizeMolecule(mol, maxIters=30)
        
        # 2. Create DataFrames
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        atoms_data = []
        for i, (sym, coord) in enumerate(zip(symbols, coords)):
            atoms_data.append({
                'molecule_name': mol_name,
                'atom_index': i,
                'atom': sym,
                'x': coord[0], 'y': coord[1], 'z': coord[2]
            })
        atoms_df = pd.DataFrame(atoms_data)
        
        # Bonds (Candidates)
        candidates = []
        num_atoms = mol.GetNumAtoms()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                path = Chem.GetShortestPath(mol, i, j)
                dist = len(path) - 1
                if 1 <= dist <= 3:
                    a1 = mol.GetAtomWithIdx(i)
                    a2 = mol.GetAtomWithIdx(j)
                    s1, s2 = a1.GetSymbol(), a2.GetSymbol()
                    
                    # Follow Kaggle convention: H comes first if present
                    if s1 == 'H' or s2 == 'H':
                        if s2 == 'H':
                            s1, s2 = s2, s1  # Swap so H is first
                    else:
                        # No H: sort alphabetically
                        if s1 > s2:
                            s1, s2 = s2, s1
                    
                    bond_type = f"{dist}J{s1}{s2}"
                    
                    candidates.append({
                        'id': -1,
                        'molecule_name': mol_name,
                        'atom_index_0': i,
                        'atom_index_1': j,
                        'type': bond_type,
                        'scalar_coupling_constant': 0.0
                    })
        
        if not candidates:
            return {'index': idx, 'status': 'no_candidates'}
        
        bonds_df = pd.DataFrame(candidates)
        
        # 3. Run Pipeline Functions
        structure_dict = make_structure_dict(atoms_df)
        enhance_structure_dict(structure_dict, skip_pybel=True)
        
        enhance_atoms(atoms_df, structure_dict)
        enhance_bonds(bonds_df, structure_dict)
        
        triplets_df = make_triplets([mol_name], structure_dict)
        quads_df = make_quadruplets([mol_name], structure_dict)
        
        # 4. Apply Embeddings and Scaling
        add_embedding(atoms_df, bonds_df, triplets_df, quads_df, embeddings=embeddings)
        add_scaling(bonds_df, means, stds)
        
        # 5. Create Tensors
        tensors = create_dataset(atoms_df, bonds_df, triplets_df, quads_df, labeled=True)
        
        result = {
            'index': idx,
            'tensors': tensors,
            'candidates': bonds_df[['atom_index_0', 'atom_index_1', 'type']].to_dict('records')
        }
        
        # 立即保存到缓存并释放内存
        if cache_dir:
            cache_path = os.path.join(cache_dir, f'mol_{idx}.pkl.gz')
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            # 清理大对象
            del mol, atoms_df, bonds_df, triplets_df, quads_df, tensors, structure_dict
            
            return {'index': idx, 'status': 'success'}
        
        return result

    except Exception as e:
        return {'index': idx, 'status': 'error', 'error_msg': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Predict Scalar Coupling Constants (J) from SMILES.")
    parser.add_argument("--workers", type=int, default=4,  # 本地默认4核
                        help="Number of parallel worker processes")
    parser.add_argument('--input', type=str, default='../filtered_test_dataset.csv', help='Input CSV')
    parser.add_argument('--output', type=str, default='../predicted_couplings.csv', help='Output CSV')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for GPU inference')
    parser.add_argument('--model', type=str, default='model_A', help='Model name')
    parser.add_argument('--chunk_size', type=int, default=50, help='Process molecules in chunks to save memory')
    parser.add_argument('--max_mols', type=int, default=None,
                        help='Optional limit on number of molecules to process (for quick validation)')
    args = parser.parse_args()

    print(f"Using {args.workers} worker processes for parallel execution.")
    
    # 1. Load Data
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} molecules from {args.input}")

    if args.max_mols is not None:
        df = df.head(args.max_mols)
        print(f"Limiting run to first {len(df)} molecules (max_mols={args.max_mols})")

    # 2. Build/Load Embeddings
    embeddings, means, stds = build_embeddings_and_scaling()
    
    # Create cache directory
    cache_dir = os.path.join(os.path.dirname(args.output), 'cache_features')
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    # 3. 分块并行处理（关键改进）
    print("Checking cache for existing files...")
    existing_indices = set()
    for f in os.listdir(cache_dir):
        if f.startswith('mol_') and f.endswith('.pkl.gz'):
            try:
                idx = int(f.split('_')[1].split('.')[0])
                existing_indices.add(idx)
            except:
                pass
    
    all_indices = list(range(len(df)))
    todo_indices = [i for i in all_indices if i not in existing_indices]
    
    print(f"Total molecules: {len(df)}")
    print(f"Already processed: {len(existing_indices)}")
    print(f"Remaining to process: {len(todo_indices)}")
    
    if todo_indices:
        print(f"Starting feature extraction in chunks of {args.chunk_size}...")
        
        # 分块处理，避免内存堆积
        num_chunks = int(np.ceil(len(todo_indices) / args.chunk_size))
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * args.chunk_size
            end = min((chunk_idx + 1) * args.chunk_size, len(todo_indices))
            chunk_indices = todo_indices[start:end]
            
            print(f"\n--- Chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_indices)} molecules) ---")
            
            tasks = [(i, df.iloc[i]) for i in chunk_indices]
            worker_func = functools.partial(process_single_molecule, 
                                          embeddings=embeddings, 
                                          means=means, 
                                          stds=stds, 
                                          cache_dir=cache_dir)
            
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # 立即迭代，不存储结果
                for result in tqdm(executor.map(worker_func, tasks), 
                                 total=len(tasks), 
                                 desc=f"Chunk {chunk_idx+1}"):
                    pass  # 结果已经在 worker 中保存到缓存
            
            # 每个 chunk 处理完后强制垃圾回收
            import gc
            gc.collect()
    else:
        print("All molecules already processed.")

    # 4. Batch Inference
    print("\nInitializing Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        predictor = Predictor(model_name=args.model, device=str(device))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Running Inference...")
    final_records = []
    
    # 分批加载缓存并推理
    for i in tqdm(range(0, len(df), args.batch_size), desc="Inference"):
        batch_indices = range(i, min(i + args.batch_size, len(df)))
        batch_items = []
        
        for idx in batch_indices:
            cache_path = os.path.join(cache_dir, f'mol_{idx}.pkl.gz')
            if os.path.exists(cache_path):
                try:
                    with gzip.open(cache_path, 'rb') as f:
                        res = pickle.load(f)
                        if 'error' not in res and 'status' not in res:
                            batch_items.append(res)
                except Exception:
                    pass
        
        if not batch_items:
            continue
            
        # Stack tensors
        try:
            batched_tensors = []
            num_components = len(batch_items[0]['tensors'])
            
            for comp_idx in range(num_components):
                t = torch.cat([item['tensors'][comp_idx] for item in batch_items], dim=0)
                batched_tensors.append(t)
                
            # Predict
            preds_scaled, x_bond, y_info = predictor.predict(tuple(batched_tensors))
            
            preds_np = preds_scaled.cpu().numpy()
            y_info_np = y_info.cpu().numpy()
            
            for b_idx, item in enumerate(batch_items):
                idx = item['index']
                candidates = item['candidates']
                original_row = df.iloc[idx]
                
                mol_preds = preds_np[b_idx]
                mol_predict_mask = y_info_np[b_idx, :, 3]
                
                valid_indices = np.where(mol_predict_mask > 0.5)[0]
                valid_preds = mol_preds[valid_indices]
                
                length = min(len(valid_preds), len(candidates))
                valid_preds = valid_preds[:length]
                candidates = candidates[:length]
                
                for r, pred_val in enumerate(valid_preds):
                    cand = candidates[r]
                    final_records.append({
                        'smiles': original_row['smiles'],
                        'atom_index_0': cand['atom_index_0'],
                        'atom_index_1': cand['atom_index_1'],
                        'type': cand['type'],
                        'scalar_coupling_constant': float(pred_val)
                    })
            
            # 清理批次数据
            del batched_tensors, preds_scaled, x_bond, y_info, batch_items
            
        except Exception as e:
            print(f"Batch inference error: {e}")

    # 5. Save
    result_df = pd.DataFrame(final_records)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    print(f"Total predicted bonds: {len(result_df)}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
