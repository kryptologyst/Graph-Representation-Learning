"""Training utilities for graph representation learning models."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.utils.device import get_device, move_to_device


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially restore weights.
            
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False


class Trainer:
    """Trainer for graph representation learning models."""
    
    def __init__(self, model: nn.Module, device: str = "auto", 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 optimizer: str = "adam", scheduler: bool = True,
                 early_stopping_patience: int = 10,
                 gradient_clip_norm: Optional[float] = None,
                 mixed_precision: bool = False):
        """
        Initialize trainer.
        
        Args:
            model: Model to train.
            device: Device to use.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            optimizer: Optimizer type.
            scheduler: Whether to use learning rate scheduler.
            early_stopping_patience: Early stopping patience.
            gradient_clip_norm: Gradient clipping norm.
            mixed_precision: Whether to use mixed precision.
        """
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device)
        
        # Optimizer
        if optimizer.lower() == "adam":
            self.optimizer = Adam(model.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay)
        elif optimizer.lower() == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
            
        # Scheduler
        self.scheduler = None
        if scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', 
                                             patience=5, factor=0.5)
            
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Other settings
        self.gradient_clip_norm = gradient_clip_norm
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def train_epoch(self, data: torch.Tensor, edge_index: torch.Tensor,
                   labels: Optional[torch.Tensor] = None) -> float:
        """
        Train for one epoch.
        
        Args:
            data: Input data.
            edge_index: Edge indices.
            labels: Optional labels for supervised learning.
            
        Returns:
            Training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = move_to_device(data, self.device)
        edge_index = move_to_device(edge_index, self.device)
        if labels is not None:
            labels = move_to_device(labels, self.device)
            
        if self.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                if hasattr(self.model, 'loss') and labels is None:
                    # Unsupervised learning (DGI, GraphCL, GRACE)
                    if hasattr(self.model, 'corrupt_features'):
                        # DGI
                        x_corrupted = self.model.corrupt_features(data)
                        loss = self.model.loss(data, edge_index, x_corrupted)
                    else:
                        # GraphCL or GRACE
                        loss = self.model.contrastive_loss(data, edge_index)
                else:
                    # Supervised learning or forward pass
                    output = self.model(data, edge_index)
                    if labels is not None:
                        loss = F.cross_entropy(output, labels)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                        
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.gradient_clip_norm)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if hasattr(self.model, 'loss') and labels is None:
                # Unsupervised learning
                if hasattr(self.model, 'corrupt_features'):
                    # DGI
                    x_corrupted = self.model.corrupt_features(data)
                    loss = self.model.loss(data, edge_index, x_corrupted)
                else:
                    # GraphCL or GRACE
                    loss = self.model.contrastive_loss(data, edge_index)
            else:
                # Supervised learning
                output = self.model(data, edge_index)
                if labels is not None:
                    loss = F.cross_entropy(output, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                    
            loss.backward()
            
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.gradient_clip_norm)
                
            self.optimizer.step()
            
        return loss.item()
        
    def validate(self, data: torch.Tensor, edge_index: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> float:
        """
        Validate the model.
        
        Args:
            data: Input data.
            edge_index: Edge indices.
            labels: Optional labels.
            
        Returns:
            Validation loss.
        """
        self.model.eval()
        
        with torch.no_grad():
            data = move_to_device(data, self.device)
            edge_index = move_to_device(edge_index, self.device)
            if labels is not None:
                labels = move_to_device(labels, self.device)
                
            if hasattr(self.model, 'loss') and labels is None:
                # Unsupervised learning
                if hasattr(self.model, 'corrupt_features'):
                    # DGI
                    x_corrupted = self.model.corrupt_features(data)
                    loss = self.model.loss(data, edge_index, x_corrupted)
                else:
                    # GraphCL or GRACE
                    loss = self.model.contrastive_loss(data, edge_index)
            else:
                # Supervised learning
                output = self.model(data, edge_index)
                if labels is not None:
                    loss = F.cross_entropy(output, labels)
                else:
                    loss = torch.tensor(0.0, device=self.device)
                    
        return loss.item()
        
    def train(self, train_data: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
              val_data: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
              num_epochs: int = 100, verbose: bool = True) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_data: Training data tuple (features, edge_index, labels).
            val_data: Validation data tuple (features, edge_index, labels).
            num_epochs: Number of training epochs.
            verbose: Whether to show progress.
            
        Returns:
            Training history.
        """
        train_x, train_edge_index, train_labels = train_data
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        if verbose:
            pbar = tqdm(range(num_epochs), desc="Training")
        else:
            pbar = range(num_epochs)
            
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(train_x, train_edge_index, train_labels)
            history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_data is not None:
                val_x, val_edge_index, val_labels = val_data
                val_loss = self.validate(val_x, val_edge_index, val_labels)
                history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                    
                # Early stopping
                if self.early_stopping(val_loss, self.model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                    
            # Update progress bar
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                history['learning_rate'].append(lr)
                
                if val_loss is not None:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'lr': f'{lr:.6f}'
                    })
                else:
                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'lr': f'{lr:.6f}'
                    })
                    
        return history
        
    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            loss: Current loss.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'], checkpoint['loss']
