"""Data loading and preprocessing utilities."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub, Planetoid
from torch_geometric.utils import to_networkx


class GraphDataset:
    """Base class for graph datasets."""
    
    def __init__(self, name: str, data_dir: str = "data"):
        """
        Initialize graph dataset.
        
        Args:
            name: Name of the dataset.
            data_dir: Directory to store/load data.
        """
        self.name = name
        self.data_dir = data_dir
        self.graph: Optional[nx.Graph] = None
        self.node_features: Optional[torch.Tensor] = None
        self.node_labels: Optional[torch.Tensor] = None
        self.edge_index: Optional[torch.Tensor] = None
        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None
        
    def load(self) -> None:
        """Load the dataset."""
        raise NotImplementedError
        
    def get_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if self.edge_index is None or self.node_features is None:
            raise ValueError("Dataset not loaded properly")
            
        data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            y=self.node_labels,
        )
        
        if self.train_mask is not None:
            data.train_mask = self.train_mask
        if self.val_mask is not None:
            data.val_mask = self.val_mask
        if self.test_mask is not None:
            data.test_mask = self.test_mask
            
        return data


class KarateClubDataset(GraphDataset):
    """Zachary's Karate Club dataset."""
    
    def load(self) -> None:
        """Load Karate Club dataset."""
        # Load using PyG
        dataset = KarateClub()
        data = dataset[0]
        
        # Convert to NetworkX for compatibility
        self.graph = to_networkx(data, to_undirected=True)
        
        # Extract features and labels
        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index
        
        # Create train/val/test splits
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)
        
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:train_size + val_size]] = True
        self.test_mask[indices[train_size + val_size:]] = True


class CoraDataset(GraphDataset):
    """Cora citation network dataset."""
    
    def load(self) -> None:
        """Load Cora dataset."""
        dataset = Planetoid(root=self.data_dir, name='Cora')
        data = dataset[0]
        
        # Convert to NetworkX
        self.graph = to_networkx(data, to_undirected=True)
        
        # Extract features and labels
        self.node_features = data.x
        self.node_labels = data.y
        self.edge_index = data.edge_index
        
        # Use provided splits
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask


class SyntheticDataset(GraphDataset):
    """Synthetic graph dataset for testing."""
    
    def __init__(self, name: str = "synthetic", num_nodes: int = 100, 
                 num_communities: int = 4, data_dir: str = "data"):
        """
        Initialize synthetic dataset.
        
        Args:
            name: Dataset name.
            num_nodes: Number of nodes.
            num_communities: Number of communities.
            data_dir: Data directory.
        """
        super().__init__(name, data_dir)
        self.num_nodes = num_nodes
        self.num_communities = num_communities
        
    def load(self) -> None:
        """Generate synthetic dataset using stochastic block model."""
        # Generate SBM graph
        sizes = [self.num_nodes // self.num_communities] * self.num_communities
        p_in = 0.3  # Within-community edge probability
        p_out = 0.05  # Between-community edge probability
        
        # Create probability matrix
        probs = np.full((self.num_communities, self.num_communities), p_out)
        np.fill_diagonal(probs, p_in)
        
        self.graph = nx.stochastic_block_model(sizes, probs, seed=42)
        
        # Generate node features (random)
        self.node_features = torch.randn(self.num_nodes, 16)
        
        # Generate node labels (community assignments)
        self.node_labels = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx = 0
        for i, size in enumerate(sizes):
            self.node_labels[node_idx:node_idx + size] = i
            node_idx += size
            
        # Convert to edge index
        edge_list = list(self.graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list).t().contiguous()
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        self.edge_index = edge_index
        
        # Create train/val/test splits
        indices = torch.randperm(self.num_nodes)
        train_size = int(0.7 * self.num_nodes)
        val_size = int(0.15 * self.num_nodes)
        
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:train_size + val_size]] = True
        self.test_mask[indices[train_size + val_size:]] = True


def get_dataset(name: str, data_dir: str = "data", **kwargs) -> GraphDataset:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name.
        data_dir: Data directory.
        **kwargs: Additional arguments for dataset initialization.
        
    Returns:
        GraphDataset instance.
    """
    if name.lower() == "karate":
        dataset = KarateClubDataset(name, data_dir)
    elif name.lower() == "cora":
        dataset = CoraDataset(name, data_dir)
    elif name.lower() == "synthetic":
        dataset = SyntheticDataset(name, data_dir=data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    dataset.load()
    return dataset


def generate_random_walks(graph: nx.Graph, num_walks: int = 10, 
                         walk_length: int = 20, seed: int = 42) -> List[List[str]]:
    """
    Generate random walks on the graph.
    
    Args:
        graph: NetworkX graph.
        num_walks: Number of walks per node.
        walk_length: Length of each walk.
        seed: Random seed.
        
    Returns:
        List of random walks.
    """
    random.seed(seed)
    walks = []
    nodes = list(graph.nodes())
    
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [str(node)]
            current_node = node
            
            while len(walk) < walk_length:
                neighbors = list(graph.neighbors(current_node))
                if not neighbors:
                    break
                    
                next_node = random.choice(neighbors)
                walk.append(str(next_node))
                current_node = next_node
                
            walks.append(walk)
            
    return walks
