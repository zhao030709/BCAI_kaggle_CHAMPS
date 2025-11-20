"""
Training script for Set Transformer J-coupling prediction.

Usage:
    python train_set_transformer.py --config config.json
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from set_transformer_model import create_model
from set_transformer_loss import create_loss_function
from set_transformer_data import load_pseudo_labels, create_dataloaders


class Trainer:
    """
    Trainer class for Set Transformer.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create model
        print("Creating model...")
        self.model = create_model(config['model'])
        self.model.to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create loss function
        self.criterion = create_loss_function(
            loss_type=config.get('loss_type', 'hungarian'),
            **config.get('loss_params', {})
        )
        
        # Create optimizer
        optimizer_name = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 1e-4)
        
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.01))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 5),
            verbose=True
        )
        
        # Logging
        self.writer = SummaryWriter(config.get('log_dir', 'runs/set_transformer'))
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = config.get('early_stopping_patience', 20)
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_metrics = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = self.model(batch)
            
            # Compute loss
            loss, metrics = self.criterion(
                pred_j=output['j_values'],
                pred_type_logits=output['type_logits'],
                target_j=batch['target_j'],
                target_types=batch['target_types'],
                pred_mask=batch['pred_mask'],
                target_mask=batch['target_mask']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        
        for batch in tqdm(val_loader, desc="Validation"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = self.model(batch)
            
            # Compute loss
            loss, metrics = self.criterion(
                pred_j=output['j_values'],
                pred_type_logits=output['type_logits'],
                target_j=batch['target_j'],
                target_types=batch['target_types'],
                pred_mask=batch['pred_mask'],
                target_mask=batch['target_mask']
            )
            
            # Accumulate metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        
        # Average metrics
        avg_loss = total_loss / len(val_loader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def save_checkpoint(self, path, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def train(self, train_loader, val_loader=None):
        """Main training loop."""
        print("="*60)
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Training batches: {len(train_loader)}")
        if val_loader:
            print(f"Validation batches: {len(val_loader)}")
        print("="*60)
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Log training metrics
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            
            print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
            print(f"  J-MAE: {train_metrics.get('j_mae', 0):.4f} Hz")
            print(f"  Type Acc: {train_metrics.get('type_accuracy', 0):.3f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
                
                # Log validation metrics
                for k, v in val_metrics.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)
                
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val J-MAE: {val_metrics.get('j_mae', 0):.4f} Hz")
                print(f"  Val Type Acc: {val_metrics.get('type_accuracy', 0):.3f}")
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])
                
                # Early stopping and checkpointing
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    
                    # Save best model
                    checkpoint_path = os.path.join(self.config['checkpoint_dir'], 
                                                  f'checkpoint_epoch{epoch}.pt')
                    self.save_checkpoint(checkpoint_path, epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.max_patience:
                        print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                        break
            
            # Periodic checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                checkpoint_path = os.path.join(self.config['checkpoint_dir'], 
                                              f'checkpoint_epoch{epoch}.pt')
                self.save_checkpoint(checkpoint_path, epoch)
        
        print("\n✅ Training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/set_transformer_config.json',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='pseudo_labeled_dataset.csv',
                       help='Path to pseudo-labeled dataset')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            'model': {
                'd_model': 256,
                'n_encoder_layers': 6,
                'n_heads': 8,
                'd_ff': 1024,
                'n_coupling_types': 8,
                'max_atoms': 100,
                'dropout': 0.1
            },
            'loss_type': 'hungarian',
            'loss_params': {
                'alpha': 1.0,
                'beta': 0.5
            },
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'batch_size': 16,
            'epochs': 100,
            'grad_clip': 1.0,
            'scheduler_patience': 5,
            'early_stopping_patience': 20,
            'save_every': 10,
            'checkpoint_dir': 'checkpoints/set_transformer',
            'log_dir': 'runs/set_transformer'
        }
        
        # Save default config
        os.makedirs('config', exist_ok=True)
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created default config at {args.config}")
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    smiles_list, coupling_data = load_pseudo_labels(args.data)
    print(f"Loaded {len(smiles_list)} molecules")
    print(f"Total couplings: {sum(len(c) for c in coupling_data)}")
    
    # Split train/val
    n_samples = len(smiles_list)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_smiles = [smiles_list[i] for i in train_indices]
    train_couplings = [coupling_data[i] for i in train_indices]
    val_smiles = [smiles_list[i] for i in val_indices]
    val_couplings = [coupling_data[i] for i in val_indices]
    
    print(f"Train: {len(train_smiles)} molecules")
    print(f"Val: {len(val_smiles)} molecules")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_smiles, train_couplings,
        val_smiles, val_couplings,
        batch_size=config['batch_size'],
        max_atoms=config['model']['max_atoms']
    )
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
