"""Tests for graph representation learning models."""

import pytest
import torch
import networkx as nx
from src.data.dataset import get_dataset, SyntheticDataset
from src.models.deepwalk import DeepWalk
from src.models.dgi import DGI
from src.models.graphcl import GraphCL
from src.models.grace import GRACE
from src.utils.device import set_seed


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    set_seed(42)
    dataset = SyntheticDataset(num_nodes=20, num_communities=2)
    dataset.load()
    return dataset


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G


def test_synthetic_dataset(synthetic_dataset):
    """Test synthetic dataset creation."""
    assert synthetic_dataset.graph.number_of_nodes() == 20
    assert synthetic_dataset.node_features.shape[0] == 20
    assert synthetic_dataset.node_labels.shape[0] == 20
    assert synthetic_dataset.edge_index.shape[1] > 0


def test_deepwalk(simple_graph):
    """Test DeepWalk implementation."""
    model = DeepWalk(embedding_dim=16, num_walks=2, walk_length=5)
    model.fit(simple_graph)
    
    embeddings = model.get_embeddings()
    assert len(embeddings) == simple_graph.number_of_nodes()
    assert embeddings[0].shape[0] == 16


def test_dgi(synthetic_dataset):
    """Test DGI implementation."""
    data = synthetic_dataset.get_pyg_data()
    model = DGI(
        input_dim=data.x.shape[1],
        hidden_dim=16,
        output_dim=16,
        num_layers=2
    )
    
    # Test forward pass
    embeddings = model(data.x, data.edge_index)
    assert embeddings.shape == (data.x.shape[0], 16)
    
    # Test loss computation
    x_corrupted = model.corrupt_features(data.x)
    loss = model.loss(data.x, data.edge_index, x_corrupted)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_graphcl(synthetic_dataset):
    """Test GraphCL implementation."""
    data = synthetic_dataset.get_pyg_data()
    model = GraphCL(
        input_dim=data.x.shape[1],
        hidden_dim=16,
        output_dim=16,
        num_layers=2
    )
    
    # Test forward pass
    embeddings = model(data.x, data.edge_index)
    assert embeddings.shape == (data.x.shape[0], 16)
    
    # Test contrastive loss
    loss = model.contrastive_loss(data.x, data.edge_index)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_grace(synthetic_dataset):
    """Test GRACE implementation."""
    data = synthetic_dataset.get_pyg_data()
    model = GRACE(
        input_dim=data.x.shape[1],
        hidden_dim=16,
        output_dim=16,
        num_layers=2
    )
    
    # Test forward pass
    embeddings = model(data.x, data.edge_index)
    assert embeddings.shape == (data.x.shape[0], 16)
    
    # Test contrastive loss
    loss = model.contrastive_loss(data.x, data.edge_index)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0


def test_model_training(synthetic_dataset):
    """Test model training process."""
    data = synthetic_dataset.get_pyg_data()
    model = DGI(
        input_dim=data.x.shape[1],
        hidden_dim=16,
        output_dim=16,
        num_layers=2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop
    for epoch in range(3):
        optimizer.zero_grad()
        x_corrupted = model.corrupt_features(data.x)
        loss = model.loss(data.x, data.edge_index, x_corrupted)
        loss.backward()
        optimizer.step()
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


def test_device_handling():
    """Test device handling utilities."""
    from src.utils.device import get_device, set_seed
    
    # Test seed setting
    set_seed(42)
    
    # Test device selection
    device = get_device("cpu")
    assert device.type == "cpu"
    
    device = get_device("auto")
    assert device.type in ["cpu", "cuda", "mps"]


def test_config_loading():
    """Test configuration loading."""
    from src.utils.config import Config, load_config
    
    # Test default config
    config = Config()
    assert config.data.dataset_name == "karate"
    assert config.model.model_type == "deepwalk"
    
    # Test config conversion
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "data" in config_dict
    assert "model" in config_dict


if __name__ == "__main__":
    pytest.main([__file__])
