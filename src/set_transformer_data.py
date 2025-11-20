"""
Data loader for Set Transformer training.

Converts SMILES + J-coupling data into graph representations.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Optional
import pickle


class MolecularGraphDataset(Dataset):
    """
    Dataset for molecular graphs with J-coupling targets.
    
    Each sample contains:
    - Molecular graph (atoms, bonds, distances)
    - Target J-coupling set {(j_value, type)}
    """
    def __init__(self, 
                 smiles_list: List[str],
                 coupling_data: List[List[Dict]],
                 max_atoms: int = 100):
        """
        Args:
            smiles_list: List of SMILES strings
            coupling_data: List of coupling lists, where each coupling is
                          {'j_value': float, 'type': str, 'atom_0': int, 'atom_1': int}
            max_atoms: Maximum number of atoms (for padding)
        """
        self.smiles_list = smiles_list
        self.coupling_data = coupling_data
        self.max_atoms = max_atoms
        
        # Coupling type mapping
        self.coupling_types = ['1JHC', '2JHC', '3JHC', '1JHH', '2JHH', '3JHH', '2JHN', '3JHN']
        self.type_to_idx = {t: i for i, t in enumerate(self.coupling_types)}
        
        # Atom type vocabulary (atomic numbers)
        self.atom_vocab = list(range(1, 120))  # H=1 to element 119
        
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def _smiles_to_graph(self, smiles: str) -> Dict:
        """
        Convert SMILES to graph representation.
        
        Returns:
            Dictionary with graph data
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens for complete graph
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()
        
        # Atom features
        atom_types = []
        atom_features = []
        
        for atom in mol.GetAtoms():
            # Atom type (atomic number)
            atom_types.append(atom.GetAtomicNum())
            
            # Additional features
            features = [
                atom.GetDegree() / 4.0,  # Normalized degree
                atom.GetTotalValence() / 6.0,
                atom.GetFormalCharge() / 2.0,
                float(atom.GetIsAromatic()),
                atom.GetHybridization().real / 6.0,  # SP, SP2, SP3, etc.
                atom.GetTotalNumHs() / 4.0,
                float(atom.IsInRing()),
                atom.GetNumRadicalElectrons() / 2.0,
                float(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
                0.0  # Placeholder for additional feature
            ]
            atom_features.append(features)
        
        atom_types = np.array(atom_types, dtype=np.int64)
        atom_features = np.array(atom_features, dtype=np.float32)
        
        # Graph distance matrix (shortest path)
        dist_matrix = Chem.GetDistanceMatrix(mol)
        dist_matrix = np.clip(dist_matrix, 0, 4).astype(np.int64)  # Clip to 4+ hops
        
        # Pair features (for decoder)
        pair_features = np.zeros((n_atoms, n_atoms, 5), dtype=np.float32)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                
                # Distance feature
                pair_features[i, j, 0] = dist_matrix[i, j] / 4.0
                
                # Bond features (if bond exists)
                bond = mol.GetBondBetweenAtoms(int(i), int(j))
                if bond is not None:
                    bond_type = bond.GetBondTypeAsDouble()
                    pair_features[i, j, 1] = bond_type / 3.0  # Single=1, Double=2, Triple=3
                    pair_features[i, j, 2] = float(bond.GetIsConjugated())
                    pair_features[i, j, 3] = float(bond.GetIsAromatic())
                    pair_features[i, j, 4] = float(bond.IsInRing())
        
        return {
            'n_atoms': n_atoms,
            'atom_types': atom_types,
            'atom_features': atom_features,
            'graph_dist': dist_matrix,
            'pair_features': pair_features
        }
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one sample.
        
        Returns:
            Dictionary with graph and target data
        """
        smiles = self.smiles_list[idx]
        couplings = self.coupling_data[idx]
        
        # Convert SMILES to graph
        graph_data = self._smiles_to_graph(smiles)
        n_atoms = graph_data['n_atoms']
        
        # Extract targets
        target_j_values = []
        target_types = []
        
        for coupling in couplings:
            target_j_values.append(coupling['j_value'])
            type_str = coupling['type']
            target_types.append(self.type_to_idx.get(type_str, 0))
        
        # Padding
        atom_types_padded = np.zeros(self.max_atoms, dtype=np.int64)
        atom_features_padded = np.zeros((self.max_atoms, 10), dtype=np.float32)
        graph_dist_padded = np.zeros((self.max_atoms, self.max_atoms), dtype=np.int64)
        pair_features_padded = np.zeros((self.max_atoms, self.max_atoms, 5), dtype=np.float32)
        
        atom_types_padded[:n_atoms] = graph_data['atom_types']
        atom_features_padded[:n_atoms] = graph_data['atom_features']
        graph_dist_padded[:n_atoms, :n_atoms] = graph_data['graph_dist']
        pair_features_padded[:n_atoms, :n_atoms] = graph_data['pair_features']
        
        # Masks
        atom_mask = np.zeros(self.max_atoms, dtype=bool)
        atom_mask[:n_atoms] = True
        
        pair_mask = np.zeros((self.max_atoms, self.max_atoms), dtype=bool)
        pair_mask[:n_atoms, :n_atoms] = True
        
        return {
            'smiles': smiles,
            'atom_types': torch.from_numpy(atom_types_padded),
            'atom_features': torch.from_numpy(atom_features_padded),
            'graph_dist': torch.from_numpy(graph_dist_padded),
            'pair_features': torch.from_numpy(pair_features_padded),
            'atom_mask': torch.from_numpy(atom_mask),
            'pair_mask': torch.from_numpy(pair_mask),
            'target_j': torch.tensor(target_j_values, dtype=torch.float32),
            'target_types': torch.tensor(target_types, dtype=torch.long),
            'n_targets': len(target_j_values)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length targets.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary
    """
    batch_size = len(batch)
    max_atoms = batch[0]['atom_types'].size(0)
    max_targets = max(sample['n_targets'] for sample in batch)
    
    # Initialize tensors
    atom_types = torch.stack([sample['atom_types'] for sample in batch])
    atom_features = torch.stack([sample['atom_features'] for sample in batch])
    graph_dist = torch.stack([sample['graph_dist'] for sample in batch])
    pair_features = torch.stack([sample['pair_features'] for sample in batch])
    atom_mask = torch.stack([sample['atom_mask'] for sample in batch])
    pair_mask = torch.stack([sample['pair_mask'] for sample in batch])
    
    # Pad targets
    target_j = torch.zeros(batch_size, max_targets, dtype=torch.float32)
    target_types = torch.zeros(batch_size, max_targets, dtype=torch.long)
    target_mask = torch.zeros(batch_size, max_targets, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        n_targets = sample['n_targets']
        if n_targets > 0:
            target_j[i, :n_targets] = sample['target_j']
            target_types[i, :n_targets] = sample['target_types']
            target_mask[i, :n_targets] = True
    
    # For predictions, use all valid pairs
    max_pairs = max_atoms * max_atoms
    pred_mask = pair_mask.view(batch_size, -1)
    
    return {
        'atom_types': atom_types,
        'atom_features': atom_features,
        'graph_dist': graph_dist,
        'pair_features': pair_features,
        'atom_mask': atom_mask,
        'pair_mask': pair_mask,
        'pred_mask': pred_mask,
        'target_j': target_j,
        'target_types': target_types,
        'target_mask': target_mask,
        'smiles': [sample['smiles'] for sample in batch]
    }


def load_pseudo_labels(pseudo_label_csv: str) -> Tuple[List[str], List[List[Dict]]]:
    """
    Load pseudo-labeled dataset.
    
    Args:
        pseudo_label_csv: Path to pseudo_labeled_dataset.csv
    
    Returns:
        (smiles_list, coupling_data_list)
    """
    df = pd.read_csv(pseudo_label_csv)
    
    # 检测列名并适配
    j_col = 'j_value' if 'j_value' in df.columns else 'scalar_coupling_constant'
    
    # Group by SMILES
    grouped = df.groupby('smiles')
    
    smiles_list = []
    coupling_data_list = []
    
    for smiles, group in grouped:
        smiles_list.append(smiles)
        
        couplings = []
        for _, row in group.iterrows():
            couplings.append({
                'j_value': row[j_col],
                'type': row['type'],
                'atom_0': row['atom_index_0'],
                'atom_1': row['atom_index_1']
            })
        
        coupling_data_list.append(couplings)
    
    return smiles_list, coupling_data_list


def create_dataloaders(
    train_smiles: List[str],
    train_couplings: List[List[Dict]],
    val_smiles: Optional[List[str]] = None,
    val_couplings: Optional[List[List[Dict]]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    max_atoms: int = 100
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_smiles: Training SMILES list
        train_couplings: Training coupling data
        val_smiles: Validation SMILES list (optional)
        val_couplings: Validation coupling data (optional)
        batch_size: Batch size
        num_workers: Number of dataloader workers
        max_atoms: Maximum atoms for padding
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = MolecularGraphDataset(
        train_smiles, train_couplings, max_atoms
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = None
    if val_smiles is not None and val_couplings is not None:
        val_dataset = MolecularGraphDataset(
            val_smiles, val_couplings, max_atoms
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test data loader
    print("Testing data loader...")
    
    # Dummy data
    smiles_list = [
        'CCO',  # Ethanol
        'CC(C)O',  # Isopropanol
        'c1ccccc1'  # Benzene
    ]
    
    coupling_data = [
        [
            {'j_value': 7.5, 'type': '3JHH', 'atom_0': 0, 'atom_1': 1},
            {'j_value': 140.0, 'type': '1JHC', 'atom_0': 0, 'atom_1': 5}
        ],
        [
            {'j_value': 7.0, 'type': '3JHH', 'atom_0': 0, 'atom_1': 2},
            {'j_value': 5.0, 'type': '3JHC', 'atom_0': 1, 'atom_1': 5}
        ],
        [
            {'j_value': 7.8, 'type': '3JHH', 'atom_0': 1, 'atom_1': 3}
        ]
    ]
    
    dataset = MolecularGraphDataset(smiles_list, coupling_data, max_atoms=50)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    for batch in loader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Batch size: {batch['atom_types'].size(0)}")
        print(f"Atom types shape: {batch['atom_types'].shape}")
        print(f"Target J shape: {batch['target_j'].shape}")
        print(f"Target types shape: {batch['target_types'].shape}")
        print(f"Number of targets per sample: {batch['target_mask'].sum(dim=1).tolist()}")
        break
    
    print("\n✅ Data loader test passed!")
