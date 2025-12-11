"""GRACE (Graph Contrastive Learning with Adaptive Augmentation) implementation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops, dropout_adj, dropout_edge


class GRACEEncoder(nn.Module):
    """Graph encoder for GRACE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize GRACE encoder.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x


class GRACE(nn.Module):
    """GRACE implementation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, 
                 temperature: float = 0.1, edge_dropout: float = 0.1,
                 feature_dropout: float = 0.1):
        """
        Initialize GRACE model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            temperature: Temperature parameter for contrastive loss.
            edge_dropout: Edge dropout rate for augmentation.
            feature_dropout: Feature dropout rate for augmentation.
        """
        super().__init__()
        self.temperature = temperature
        self.edge_dropout = edge_dropout
        self.feature_dropout = feature_dropout
        
        # Encoder
        self.encoder = GRACEEncoder(input_dim, hidden_dim, output_dim, 
                                  num_layers, dropout)
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        return self.encoder(x, edge_index)
        
    def get_graph_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get graph-level embedding.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Graph embedding.
        """
        node_embeddings = self.encoder(x, edge_index)
        return global_mean_pool(node_embeddings, batch=None)
        
    def augment(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of the graph.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Two augmented views (x1, edge_index1), (x2, edge_index2).
        """
        # View 1: Edge dropout
        edge_index1, _ = dropout_edge(edge_index, p=self.edge_dropout, 
                                     training=self.training)
        edge_index1, _ = add_self_loops(edge_index1, num_nodes=x.size(0))
        x1 = x.clone()
        
        # View 2: Feature dropout
        x2 = x.clone()
        mask = torch.rand(x.size(0), x.size(1)) < self.feature_dropout
        x2[mask] = 0
        edge_index2, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        return (x1, edge_index1), (x2, edge_index2)
        
    def contrastive_loss(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute GRACE contrastive loss.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Contrastive loss.
        """
        # Generate two views
        (x1, edge_index1), (x2, edge_index2) = self.augment(x, edge_index)
        
        # Get embeddings
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        
        # Project embeddings
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Labels for contrastive learning
        labels = torch.arange(z1.size(0)).to(z1.device)
        
        # Symmetric loss
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss1 + loss2) / 2
        
    def info_nce_loss(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss (alternative to contrastive loss).
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            InfoNCE loss.
        """
        # Generate two views
        (x1, edge_index1), (x2, edge_index2) = self.augment(x, edge_index)
        
        # Get embeddings
        z1 = self.encoder(x1, edge_index1)
        z2 = self.encoder(x2, edge_index2)
        
        # Project embeddings
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute positive pairs similarity
        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature
        
        # Compute negative pairs similarity
        neg_sim = torch.mm(z1, z2.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(z1.size(0)).to(z1.device)
        
        # InfoNCE loss
        loss = -pos_sim.mean() + torch.logsumexp(neg_sim, dim=1).mean()
        
        return loss
