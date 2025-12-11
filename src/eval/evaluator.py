"""Evaluation utilities for graph representation learning models."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch_geometric.data import Data

from src.utils.device import get_device, move_to_device


class Evaluator:
    """Evaluator for graph representation learning models."""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            device: Device to use for evaluation.
        """
        self.device = get_device(device)
        
    def evaluate_node_classification(self, model: nn.Module, 
                                   data: Data, 
                                   train_mask: torch.Tensor,
                                   val_mask: torch.Tensor,
                                   test_mask: torch.Tensor,
                                   classifier: str = "svm") -> Dict[str, float]:
        """
        Evaluate model on node classification task.
        
        Args:
            model: Trained model.
            data: Graph data.
            train_mask: Training mask.
            val_mask: Validation mask.
            test_mask: Test mask.
            classifier: Classifier to use ("svm" or "linear").
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model(data.x, data.edge_index)
            embeddings = embeddings.cpu().numpy()
            
            # Get labels
            labels = data.y.cpu().numpy()
            
            # Split data
            train_indices = train_mask.cpu().numpy()
            val_indices = val_mask.cpu().numpy()
            test_indices = test_mask.cpu().numpy()
            
            X_train = embeddings[train_indices]
            y_train = labels[train_indices]
            X_val = embeddings[val_indices]
            y_val = labels[val_indices]
            X_test = embeddings[test_indices]
            y_test = labels[test_indices]
            
            # Train classifier
            if classifier == "svm":
                clf = SVC(kernel='rbf', random_state=42)
            elif classifier == "linear":
                clf = SVC(kernel='linear', random_state=42)
            else:
                raise ValueError(f"Unknown classifier: {classifier}")
                
            clf.fit(X_train, y_train)
            
            # Predictions
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            
            # AUC (handle multi-class case)
            try:
                if len(np.unique(y_test)) == 2:
                    # Binary classification
                    y_pred_proba = clf.decision_function(X_test)
                    auc = roc_auc_score(y_test, y_pred_proba)
                else:
                    # Multi-class classification
                    y_pred_proba = clf.decision_function(X_test)
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except ValueError:
                auc = 0.0
                
            return {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'auc': auc
            }
            
    def evaluate_link_prediction(self, model: nn.Module, data: Data,
                               test_edges: torch.Tensor,
                               test_neg_edges: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on link prediction task.
        
        Args:
            model: Trained model.
            data: Graph data.
            test_edges: Test positive edges.
            test_neg_edges: Test negative edges.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model(data.x, data.edge_index)
            
            # Calculate edge scores
            def edge_score(edge_index):
                src_emb = embeddings[edge_index[0]]
                dst_emb = embeddings[edge_index[1]]
                return torch.sum(src_emb * dst_emb, dim=1)
                
            pos_scores = edge_score(test_edges)
            neg_scores = edge_score(test_neg_edges)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones(pos_scores.size(0)), 
                              torch.zeros(neg_scores.size(0))])
            
            # Calculate AUC
            auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
            
            # Calculate Average Precision
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(labels.cpu().numpy(), scores.cpu().numpy())
            
            return {
                'auc': auc,
                'ap': ap
            }
            
    def visualize_embeddings(self, model: nn.Module, data: Data,
                           labels: Optional[torch.Tensor] = None,
                           save_path: Optional[str] = None,
                           perplexity: float = 30.0,
                           n_iter: int = 1000) -> np.ndarray:
        """
        Visualize learned embeddings using t-SNE.
        
        Args:
            model: Trained model.
            data: Graph data.
            labels: Optional node labels.
            save_path: Path to save visualization.
            perplexity: t-SNE perplexity.
            n_iter: Number of t-SNE iterations.
            
        Returns:
            t-SNE projected embeddings.
        """
        model.eval()
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model(data.x, data.edge_index)
            embeddings = embeddings.cpu().numpy()
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       n_iter=n_iter, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            
            if labels is not None:
                labels = labels.cpu().numpy()
                unique_labels = np.unique(labels)
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=[colors[i]], label=f'Class {label}', alpha=0.7)
                plt.legend()
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
                
            plt.title('t-SNE Visualization of Node Embeddings')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            plt.show()
            
            return embeddings_2d
            
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Create and display confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path: Path to save confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def generate_report(self, metrics: Dict[str, float], 
                       model_name: str = "Model") -> str:
        """
        Generate evaluation report.
        
        Args:
            metrics: Dictionary of metrics.
            model_name: Name of the model.
            
        Returns:
            Formatted report string.
        """
        report = f"\n{'='*50}\n"
        report += f"Evaluation Report for {model_name}\n"
        report += f"{'='*50}\n"
        
        for metric, value in metrics.items():
            report += f"{metric.upper()}: {value:.4f}\n"
            
        report += f"{'='*50}\n"
        
        return report
        
    def save_embeddings(self, embeddings: torch.Tensor, 
                       node_ids: Optional[List[int]] = None,
                       save_path: str = "embeddings.pt") -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings: Node embeddings.
            node_ids: Optional node IDs.
            save_path: Path to save embeddings.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'embeddings': embeddings.cpu(),
            'node_ids': node_ids
        }
        
        torch.save(save_dict, save_path)
        
    def load_embeddings(self, load_path: str) -> Tuple[torch.Tensor, Optional[List[int]]]:
        """
        Load embeddings from file.
        
        Args:
            load_path: Path to load embeddings from.
            
        Returns:
            Tuple of (embeddings, node_ids).
        """
        save_dict = torch.load(load_path, map_location='cpu')
        return save_dict['embeddings'], save_dict['node_ids']
