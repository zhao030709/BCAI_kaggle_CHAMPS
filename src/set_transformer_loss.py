"""
Loss functions for Set Transformer J-coupling prediction.

Implements:
1. Hungarian Loss: Optimal bipartite matching between predictions and ground truth
2. Chamfer Distance: Symmetric nearest-neighbor distance
3. Combined loss with type classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import Dict, Tuple


class HungarianLoss(nn.Module):
    """
    Hungarian Loss for set prediction.
    
    Finds optimal assignment between predicted and ground truth sets,
    then computes loss on matched pairs.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Args:
            alpha: Weight for J-value regression loss
            beta: Weight for type classification loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, 
                pred_j: torch.Tensor,
                pred_type_logits: torch.Tensor,
                target_j: torch.Tensor,
                target_types: torch.Tensor,
                pred_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred_j: [batch, n_pred] predicted J-values
            pred_type_logits: [batch, n_pred, n_types] type logits
            target_j: [batch, n_target] ground truth J-values
            target_types: [batch, n_target] ground truth type indices
            pred_mask: [batch, n_pred] valid prediction mask
            target_mask: [batch, n_target] valid target mask
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics for logging
        """
        batch_size = pred_j.size(0)
        total_loss = 0.0
        total_matched = 0
        total_j_error = 0.0
        total_type_acc = 0.0
        
        for b in range(batch_size):
            # Get valid predictions and targets
            valid_pred_idx = pred_mask[b].nonzero(as_tuple=True)[0]
            valid_target_idx = target_mask[b].nonzero(as_tuple=True)[0]
            
            if len(valid_pred_idx) == 0 or len(valid_target_idx) == 0:
                continue
            
            pred_j_valid = pred_j[b, valid_pred_idx]  # [n_pred_valid]
            pred_type_valid = pred_type_logits[b, valid_pred_idx]  # [n_pred_valid, n_types]
            target_j_valid = target_j[b, valid_target_idx]  # [n_target_valid]
            target_type_valid = target_types[b, valid_target_idx]  # [n_target_valid]
            
            # Build cost matrix: [n_pred, n_target]
            # Cost = |j_pred - j_target| + cross_entropy(type_pred, type_target)
            j_cost = torch.abs(
                pred_j_valid.unsqueeze(1) - target_j_valid.unsqueeze(0)
            )  # [n_pred, n_target]
            
            # Type cost: negative log-likelihood
            pred_type_probs = F.softmax(pred_type_valid, dim=-1)  # [n_pred, n_types]
            type_cost = -torch.log(pred_type_probs[:, target_type_valid] + 1e-8)  # [n_pred, n_target]
            
            cost_matrix = self.alpha * j_cost + self.beta * type_cost
            
            # Hungarian algorithm (on CPU)
            cost_np = cost_matrix.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            # Compute loss on matched pairs
            matched_j_loss = F.l1_loss(
                pred_j_valid[row_ind],
                target_j_valid[col_ind],
                reduction='sum'
            )
            
            matched_type_loss = F.cross_entropy(
                pred_type_valid[row_ind],
                target_type_valid[col_ind],
                reduction='sum'
            )
            
            loss = self.alpha * matched_j_loss + self.beta * matched_type_loss
            total_loss += loss
            total_matched += len(row_ind)
            
            # Metrics
            total_j_error += matched_j_loss.item()
            pred_types_matched = pred_type_valid[row_ind].argmax(dim=-1)
            total_type_acc += (pred_types_matched == target_type_valid[col_ind]).float().sum().item()
        
        # Average over batch
        if total_matched > 0:
            avg_loss = total_loss / total_matched
            avg_j_error = total_j_error / total_matched
            avg_type_acc = total_type_acc / total_matched
        else:
            avg_loss = torch.tensor(0.0, device=pred_j.device)
            avg_j_error = 0.0
            avg_type_acc = 0.0
        
        metrics = {
            'hungarian_loss': avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss,
            'matched_pairs': total_matched,
            'j_mae': avg_j_error,
            'type_accuracy': avg_type_acc
        }
        
        return avg_loss, metrics


class ChamferLoss(nn.Module):
    """
    Chamfer Distance for set prediction.
    
    Computes symmetric nearest-neighbor distance:
    CD(X, Y) = mean(min_y ||x - y||) + mean(min_x ||x - y||)
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        """
        Args:
            alpha: Weight for J-value distance
            beta: Weight for type mismatch penalty
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self,
                pred_j: torch.Tensor,
                pred_type_logits: torch.Tensor,
                target_j: torch.Tensor,
                target_types: torch.Tensor,
                pred_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args: Same as HungarianLoss
        
        Returns:
            loss: Scalar Chamfer distance
            metrics: Dictionary of metrics
        """
        batch_size = pred_j.size(0)
        total_forward_dist = 0.0
        total_backward_dist = 0.0
        total_samples = 0
        
        for b in range(batch_size):
            # Get valid predictions and targets
            valid_pred_idx = pred_mask[b].nonzero(as_tuple=True)[0]
            valid_target_idx = target_mask[b].nonzero(as_tuple=True)[0]
            
            if len(valid_pred_idx) == 0 or len(valid_target_idx) == 0:
                continue
            
            pred_j_valid = pred_j[b, valid_pred_idx]
            pred_type_valid = pred_type_logits[b, valid_pred_idx]
            target_j_valid = target_j[b, valid_target_idx]
            target_type_valid = target_types[b, valid_target_idx]
            
            # Distance matrix: [n_pred, n_target]
            j_dist = torch.abs(
                pred_j_valid.unsqueeze(1) - target_j_valid.unsqueeze(0)
            )
            
            # Type distance: 0 if same type, 1 if different
            pred_types = pred_type_valid.argmax(dim=-1)  # [n_pred]
            type_dist = (pred_types.unsqueeze(1) != target_type_valid.unsqueeze(0)).float()
            
            dist_matrix = self.alpha * j_dist + self.beta * type_dist
            
            # Forward: for each prediction, find nearest target
            forward_dist = dist_matrix.min(dim=1)[0].mean()
            
            # Backward: for each target, find nearest prediction
            backward_dist = dist_matrix.min(dim=0)[0].mean()
            
            total_forward_dist += forward_dist
            total_backward_dist += backward_dist
            total_samples += 1
        
        if total_samples > 0:
            avg_forward = total_forward_dist / total_samples
            avg_backward = total_backward_dist / total_samples
            chamfer_dist = (avg_forward + avg_backward) / 2
        else:
            chamfer_dist = torch.tensor(0.0, device=pred_j.device)
            avg_forward = 0.0
            avg_backward = 0.0
        
        metrics = {
            'chamfer_loss': chamfer_dist.item() if isinstance(chamfer_dist, torch.Tensor) else chamfer_dist,
            'forward_dist': avg_forward.item() if isinstance(avg_forward, torch.Tensor) else avg_forward,
            'backward_dist': avg_backward.item() if isinstance(avg_backward, torch.Tensor) else avg_backward,
        }
        
        return chamfer_dist, metrics


class CombinedSetLoss(nn.Module):
    """
    Combined loss: Hungarian + Chamfer.
    
    L = lambda_h * L_hungarian + lambda_c * L_chamfer
    """
    def __init__(self, 
                 lambda_hungarian: float = 1.0,
                 lambda_chamfer: float = 0.5,
                 alpha: float = 1.0,
                 beta: float = 0.5):
        super().__init__()
        
        self.lambda_hungarian = lambda_hungarian
        self.lambda_chamfer = lambda_chamfer
        
        self.hungarian_loss = HungarianLoss(alpha, beta)
        self.chamfer_loss = ChamferLoss(alpha, beta)
        
    def forward(self,
                pred_j: torch.Tensor,
                pred_type_logits: torch.Tensor,
                target_j: torch.Tensor,
                target_types: torch.Tensor,
                pred_mask: torch.Tensor,
                target_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Returns:
            loss: Combined loss value
            metrics: Combined metrics from both losses
        """
        hung_loss, hung_metrics = self.hungarian_loss(
            pred_j, pred_type_logits, target_j, target_types,
            pred_mask, target_mask
        )
        
        cham_loss, cham_metrics = self.chamfer_loss(
            pred_j, pred_type_logits, target_j, target_types,
            pred_mask, target_mask
        )
        
        total_loss = (
            self.lambda_hungarian * hung_loss +
            self.lambda_chamfer * cham_loss
        )
        
        metrics = {
            **hung_metrics,
            **cham_metrics,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }
        
        return total_loss, metrics


def create_loss_function(loss_type: str = 'hungarian', **kwargs):
    """
    Factory function for loss functions.
    
    Args:
        loss_type: One of ['hungarian', 'chamfer', 'combined']
        **kwargs: Loss-specific parameters
    
    Returns:
        Loss function instance
    """
    if loss_type == 'hungarian':
        return HungarianLoss(**kwargs)
    elif loss_type == 'chamfer':
        return ChamferLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedSetLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test losses
    batch_size = 4
    n_pred = 10
    n_target = 8
    n_types = 8
    
    # Dummy data
    pred_j = torch.randn(batch_size, n_pred) * 5 + 5  # J-values around 5 Hz
    pred_type_logits = torch.randn(batch_size, n_pred, n_types)
    target_j = torch.randn(batch_size, n_target) * 5 + 5
    target_types = torch.randint(0, n_types, (batch_size, n_target))
    
    pred_mask = torch.ones(batch_size, n_pred).bool()
    pred_mask[0, 7:] = False  # Vary number of predictions
    target_mask = torch.ones(batch_size, n_target).bool()
    target_mask[0, 5:] = False  # Vary number of targets
    
    # Test Hungarian loss
    print("=" * 60)
    print("Testing Hungarian Loss")
    print("=" * 60)
    hung_loss = HungarianLoss()
    loss, metrics = hung_loss(pred_j, pred_type_logits, target_j, target_types,
                              pred_mask, target_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test Chamfer loss
    print("\n" + "=" * 60)
    print("Testing Chamfer Loss")
    print("=" * 60)
    cham_loss = ChamferLoss()
    loss, metrics = cham_loss(pred_j, pred_type_logits, target_j, target_types,
                              pred_mask, target_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test Combined loss
    print("\n" + "=" * 60)
    print("Testing Combined Loss")
    print("=" * 60)
    combined_loss = CombinedSetLoss()
    loss, metrics = combined_loss(pred_j, pred_type_logits, target_j, target_types,
                                  pred_mask, target_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
