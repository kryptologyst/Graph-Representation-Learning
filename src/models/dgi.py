"""Deep Graph Infomax (DGI) implementation for unsupervised graph representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize GCN encoder.
        
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


class Discriminator(nn.Module):
    """Discriminator for DGI."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 512):
        """
        Initialize discriminator.
        
        Args:
            embedding_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_embeddings: torch.Tensor, 
                graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_embeddings: Node embeddings.
            graph_embedding: Graph-level embedding.
            
        Returns:
            Discriminator scores.
        """
        # Concatenate node embeddings with graph embedding
        graph_embedding = graph_embedding.expand(node_embeddings.size(0), -1)
        combined = node_embeddings * graph_embedding
        
        return self.fc(combined)


class DGI(nn.Module):
    """Deep Graph Infomax implementation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize DGI model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.encoder = GCNEncoder(input_dim, hidden_dim, output_dim, 
                                 num_layers, dropout)
        self.discriminator = Discriminator(output_dim)
        
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
        
    def loss(self, x: torch.Tensor, edge_index: torch.Tensor, 
             x_corrupted: torch.Tensor) -> torch.Tensor:
        """
        Compute DGI loss.
        
        Args:
            x: Original node features.
            edge_index: Edge indices.
            x_corrupted: Corrupted node features.
            
        Returns:
            DGI loss.
        """
        # Positive samples
        pos_embeddings = self.encoder(x, edge_index)
        pos_graph_embedding = global_mean_pool(pos_embeddings, batch=None)
        
        # Negative samples
        neg_embeddings = self.encoder(x_corrupted, edge_index)
        
        # Discriminator scores
        pos_scores = self.discriminator(pos_embeddings, pos_graph_embedding)
        neg_scores = self.discriminator(neg_embeddings, pos_graph_embedding)
        
        # Loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        
        return pos_loss + neg_loss
        
    def corrupt_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Corrupt node features for negative sampling.
        
        Args:
            x: Original node features.
            
        Returns:
            Corrupted node features.
        """
        # Randomly shuffle features
        indices = torch.randperm(x.size(0))
        return x[indices]
