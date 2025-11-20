"""
Graph Transformer for J-coupling prediction without atom assignment.
Architecture: Graph Encoder → Set Transformer Decoder → Pairwise Prediction

Input: Molecular graph (nodes + edges)
Output: Set of J-coupling values {(j_ij, type_ij)} without atom assignment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class GraphBiasAttention(nn.Module):
    """
    Multi-Head Self-Attention with Graph Structure Bias.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + B_graph) * V
    
    B_graph: NxN bias matrix based on graph distances (1-hop, 2-hop, 3-hop...)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_nodes, d_model]
            bias: [batch_size, n_nodes, n_nodes] graph distance bias (optional)
            mask: [batch_size, n_nodes] padding mask (optional)
        
        Returns:
            [batch_size, n_nodes, d_model]
        """
        batch_size, n_nodes, _ = x.size()
        
        # Linear projections: [batch, n_nodes, d_model] -> [batch, n_nodes, n_heads, d_k]
        Q = self.W_q(x).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, n_nodes, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: [batch, n_heads, n_nodes, n_nodes]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add graph structure bias
        if bias is not None:
            # bias: [batch, n_nodes, n_nodes] -> [batch, 1, n_nodes, n_nodes]
            scores = scores + bias.unsqueeze(1)
        
        # Apply mask if provided
        if mask is not None:
            # mask: [batch, n_nodes] -> [batch, 1, 1, n_nodes]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, -1e9)
        
        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # [batch, n_heads, n_nodes, d_k] -> [batch, n_nodes, d_model]
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_nodes, self.d_model)
        
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with Graph Bias.
    
    x -> LayerNorm -> Attention(x, bias) -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = GraphBiasAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), bias, mask)
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class GraphEncoder(nn.Module):
    """
    Graph Encoder: Embed molecular graph and apply Transformer layers.
    
    Molecular graph -> Node embeddings -> Stack of Transformer layers
    """
    def __init__(self, 
                 atom_vocab_size: int = 128,
                 d_model: int = 256,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 max_atoms: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(atom_vocab_size, d_model)
        
        # Additional atom features (degree, hybridization, etc.)
        self.atom_feature_proj = nn.Linear(10, d_model)  # 10 extra features
        
        # Graph distance bias embedding (0-hop=self, 1-hop, 2-hop, 3-hop, 4+hop)
        self.distance_bias = nn.Embedding(5, n_heads)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                atom_types: torch.Tensor,
                atom_features: torch.Tensor,
                graph_dist: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_types: [batch, n_atoms] atom type IDs
            atom_features: [batch, n_atoms, n_features] extra features
            graph_dist: [batch, n_atoms, n_atoms] pairwise graph distances (0-4+)
            mask: [batch, n_atoms] valid atom mask
        
        Returns:
            [batch, n_atoms, d_model] node embeddings
        """
        # Node embeddings
        x = self.atom_embedding(atom_types)  # [batch, n_atoms, d_model]
        x = x + self.atom_feature_proj(atom_features)  # Add extra features
        
        # Graph structure bias: [batch, n_atoms, n_atoms] -> [batch, n_heads, n_atoms, n_atoms]
        # distance_bias: [5, n_heads] -> graph_dist: [batch, n_atoms, n_atoms, n_heads]
        bias = self.distance_bias(graph_dist)  # [batch, n_atoms, n_atoms, n_heads]
        bias = bias.permute(0, 3, 1, 2)  # [batch, n_heads, n_atoms, n_atoms]
        
        # But GraphBiasAttention expects [batch, n_atoms, n_atoms] per head
        # So we'll modify to broadcast: [batch, 1, n_atoms, n_atoms] and let attention handle
        # Actually, let's average over heads for simplicity
        bias = bias.mean(dim=1)  # [batch, n_atoms, n_atoms]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, bias, mask)
        
        return self.norm(x)


class PairwiseDecoder(nn.Module):
    """
    Pairwise Decoder: Predict J-coupling for all atom pairs.
    
    For each pair (i, j):
        j_ij = MLP(concat(h_i, h_j, e_ij))
    
    where e_ij encodes pair relationship (distance, bond type, etc.)
    """
    def __init__(self, d_model: int, n_coupling_types: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.n_coupling_types = n_coupling_types
        
        # Pairwise features embedding (distance, bond order, etc.)
        self.pair_feature_proj = nn.Linear(5, d_model // 2)
        
        # Prediction head: concat(h_i, h_j, e_ij) -> (j_value, coupling_type)
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2 + d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1 + n_coupling_types)  # j_value + type logits
        )
        
    def forward(self, 
                node_emb: torch.Tensor,
                pair_features: torch.Tensor,
                pair_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_emb: [batch, n_atoms, d_model]
            pair_features: [batch, n_atoms, n_atoms, n_features] pairwise features
            pair_mask: [batch, n_atoms, n_atoms] valid pair mask
        
        Returns:
            j_values: [batch, n_pairs] predicted J-coupling values
            type_logits: [batch, n_pairs, n_types] coupling type logits
        """
        batch_size, n_atoms, d_model = node_emb.size()
        
        # Expand nodes for pairwise combination
        # h_i: [batch, n_atoms, 1, d_model]
        # h_j: [batch, 1, n_atoms, d_model]
        h_i = node_emb.unsqueeze(2).expand(-1, -1, n_atoms, -1)
        h_j = node_emb.unsqueeze(1).expand(-1, n_atoms, -1, -1)
        
        # Embed pair features
        e_ij = self.pair_feature_proj(pair_features)  # [batch, n_atoms, n_atoms, d_model//2]
        
        # Concatenate: [batch, n_atoms, n_atoms, 2*d_model + d_model//2]
        pair_repr = torch.cat([h_i, h_j, e_ij], dim=-1)
        
        # Predict: [batch, n_atoms, n_atoms, 1 + n_types]
        pred = self.predictor(pair_repr)
        
        j_values = pred[..., 0]  # [batch, n_atoms, n_atoms]
        type_logits = pred[..., 1:]  # [batch, n_atoms, n_atoms, n_types]
        
        # Flatten pairs
        j_values = j_values.view(batch_size, -1)  # [batch, n_atoms^2]
        type_logits = type_logits.view(batch_size, -1, self.n_coupling_types)
        
        # Apply mask if provided
        if pair_mask is not None:
            pair_mask_flat = pair_mask.view(batch_size, -1)
            j_values = j_values * pair_mask_flat
            type_logits = type_logits * pair_mask_flat.unsqueeze(-1)
        
        return j_values, type_logits


class JCouplingSetTransformer(nn.Module):
    """
    Complete Model: Graph Encoder + Pairwise Decoder.
    
    Input: Molecular graph
    Output: Set of J-couplings {(j_ij, type_ij)} for valid pairs
    """
    def __init__(self,
                 atom_vocab_size: int = 128,
                 d_model: int = 256,
                 n_encoder_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 n_coupling_types: int = 8,
                 max_atoms: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.encoder = GraphEncoder(
            atom_vocab_size=atom_vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_atoms=max_atoms,
            dropout=dropout
        )
        
        self.decoder = PairwiseDecoder(
            d_model=d_model,
            n_coupling_types=n_coupling_types,
            dropout=dropout
        )
        
        self.coupling_type_names = [
            '1JHC', '2JHC', '3JHC', '1JHH', '2JHH', '3JHH', '2JHN', '3JHN'
        ]
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing:
                - atom_types: [batch, n_atoms]
                - atom_features: [batch, n_atoms, n_features]
                - graph_dist: [batch, n_atoms, n_atoms]
                - pair_features: [batch, n_atoms, n_atoms, n_features]
                - atom_mask: [batch, n_atoms]
                - pair_mask: [batch, n_atoms, n_atoms]
        
        Returns:
            Dictionary containing:
                - j_values: [batch, n_pairs] predicted J-values
                - type_logits: [batch, n_pairs, n_types] coupling type logits
        """
        # Encode molecular graph
        node_emb = self.encoder(
            atom_types=batch['atom_types'],
            atom_features=batch['atom_features'],
            graph_dist=batch['graph_dist'],
            mask=batch.get('atom_mask')
        )
        
        # Decode pairwise J-couplings
        j_values, type_logits = self.decoder(
            node_emb=node_emb,
            pair_features=batch['pair_features'],
            pair_mask=batch.get('pair_mask')
        )
        
        return {
            'j_values': j_values,
            'type_logits': type_logits
        }
    
    def predict(self, batch: Dict[str, torch.Tensor], 
                threshold: float = 0.5) -> List[Dict]:
        """
        Generate predictions with filtering.
        
        Args:
            batch: Input batch
            threshold: Confidence threshold for including predictions
        
        Returns:
            List of predictions per molecule:
                [{j_value: float, type: str, confidence: float}, ...]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch)
            j_values = output['j_values']
            type_probs = F.softmax(output['type_logits'], dim=-1)
            
            batch_size = j_values.size(0)
            predictions = []
            
            for b in range(batch_size):
                mol_preds = []
                
                # Get valid pairs
                if 'pair_mask' in batch:
                    valid_pairs = batch['pair_mask'][b].view(-1).cpu().numpy()
                else:
                    n_atoms = batch['atom_types'][b].size(0)
                    valid_pairs = [True] * (n_atoms * n_atoms)
                
                for i, is_valid in enumerate(valid_pairs):
                    if not is_valid:
                        continue
                    
                    j_val = j_values[b, i].item()
                    type_prob = type_probs[b, i].cpu().numpy()
                    type_idx = type_prob.argmax()
                    confidence = type_prob[type_idx]
                    
                    if confidence >= threshold:
                        mol_preds.append({
                            'j_value': j_val,
                            'type': self.coupling_type_names[type_idx],
                            'confidence': confidence
                        })
                
                predictions.append(mol_preds)
        
        return predictions


def create_model(config: Dict) -> JCouplingSetTransformer:
    """
    Factory function to create model from config.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        Initialized model
    """
    return JCouplingSetTransformer(
        atom_vocab_size=config.get('atom_vocab_size', 128),
        d_model=config.get('d_model', 256),
        n_encoder_layers=config.get('n_encoder_layers', 6),
        n_heads=config.get('n_heads', 8),
        d_ff=config.get('d_ff', 1024),
        n_coupling_types=config.get('n_coupling_types', 8),
        max_atoms=config.get('max_atoms', 100),
        dropout=config.get('dropout', 0.1)
    )


if __name__ == '__main__':
    # Test model
    config = {
        'd_model': 128,
        'n_encoder_layers': 3,
        'n_heads': 4,
        'd_ff': 512,
        'n_coupling_types': 8
    }
    
    model = create_model(config)
    
    # Dummy input
    batch = {
        'atom_types': torch.randint(0, 100, (2, 20)),  # 2 molecules, 20 atoms each
        'atom_features': torch.randn(2, 20, 10),
        'graph_dist': torch.randint(0, 5, (2, 20, 20)),
        'pair_features': torch.randn(2, 20, 20, 5),
        'atom_mask': torch.ones(2, 20).bool(),
        'pair_mask': torch.ones(2, 20, 20).bool()
    }
    
    output = model(batch)
    print(f"J-values shape: {output['j_values'].shape}")
    print(f"Type logits shape: {output['type_logits'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
