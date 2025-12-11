"""Main training script for graph representation learning."""

import argparse
import os
from typing import Dict, Any

import torch
import wandb
from omegaconf import OmegaConf

from src.data.dataset import get_dataset
from src.models.deepwalk import DeepWalk
from src.models.dgi import DGI
from src.models.graphcl import GraphCL
from src.models.grace import GRACE
from src.train.trainer import Trainer
from src.eval.evaluator import Evaluator
from src.utils.config import Config, load_config
from src.utils.device import set_seed, get_device_info


def get_model(config: Config, input_dim: int) -> torch.nn.Module:
    """
    Get model based on configuration.
    
    Args:
        config: Configuration object.
        input_dim: Input feature dimension.
        
    Returns:
        Model instance.
    """
    model_type = config.model.model_type.lower()
    
    if model_type == "deepwalk":
        return DeepWalk(
            embedding_dim=config.data.embedding_dim,
            window_size=config.data.window_size,
            num_walks=config.data.num_walks,
            walk_length=config.data.walk_length,
            seed=config.data.random_seed
        )
    elif model_type == "dgi":
        return DGI(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.data.embedding_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    elif model_type == "graphcl":
        return GraphCL(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.data.embedding_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    elif model_type == "grace":
        return GRACE(
            input_dim=input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.data.embedding_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_deepwalk(model: DeepWalk, dataset, config: Config) -> Dict[str, Any]:
    """
    Train DeepWalk model.
    
    Args:
        model: DeepWalk model.
        dataset: Dataset.
        config: Configuration.
        
    Returns:
        Training results.
    """
    # Train DeepWalk
    model.fit(dataset.graph)
    
    # Evaluate on node classification
    results = model.evaluate_node_classification(
        dataset.graph, 
        {i: dataset.node_labels[i].item() for i in range(dataset.node_labels.size(0))}
    )
    
    return results


def train_gnn_model(model: torch.nn.Module, dataset, config: Config) -> Dict[str, Any]:
    """
    Train GNN-based model (DGI, GraphCL, GRACE).
    
    Args:
        model: GNN model.
        dataset: Dataset.
        config: Configuration.
        
    Returns:
        Training results.
    """
    # Prepare data
    data = dataset.get_pyg_data()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=config.training.device,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        early_stopping_patience=config.training.early_stopping_patience,
        gradient_clip_norm=config.training.gradient_clip_norm,
        mixed_precision=config.training.mixed_precision
    )
    
    # Training data
    train_data = (data.x, data.edge_index, None)  # Unsupervised learning
    val_data = None  # No validation for unsupervised learning
    
    # Train model
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config.training.num_epochs
    )
    
    # Evaluate model
    evaluator = Evaluator(device=config.training.device)
    
    # Node classification evaluation
    metrics = evaluator.evaluate_node_classification(
        model=model,
        data=data,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask
    )
    
    return {
        'metrics': metrics,
        'history': history
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train graph representation learning models")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="karate", help="Dataset name")
    parser.add_argument("--model", type=str, default="deepwalk", help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = Config()
        
    # Override with command line arguments
    config.data.dataset_name = args.dataset
    config.model.model_type = args.model
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.device = args.device
    config.data.random_seed = args.seed
    
    # Set random seed
    set_seed(config.data.random_seed)
    
    # Initialize wandb if requested
    if args.wandb and config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            config=config.to_dict(),
            name=f"{args.model}_{args.dataset}"
        )
    
    # Print device info
    device_info = get_device_info()
    print("Device Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print(f"\nLoading dataset: {config.data.dataset_name}")
    dataset = get_dataset(config.data.dataset_name, config.data.data_dir)
    
    print(f"Dataset info:")
    print(f"  Nodes: {dataset.graph.number_of_nodes()}")
    print(f"  Edges: {dataset.graph.number_of_edges()}")
    print(f"  Features: {dataset.node_features.shape[1] if dataset.node_features is not None else 'None'}")
    print(f"  Classes: {len(torch.unique(dataset.node_labels)) if dataset.node_labels is not None else 'None'}")
    
    # Get model
    input_dim = dataset.node_features.shape[1] if dataset.node_features is not None else 1
    model = get_model(config, input_dim)
    
    print(f"\nModel: {config.model.model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train model
    print(f"\nTraining {config.model.model_type}...")
    
    if config.model.model_type.lower() == "deepwalk":
        results = train_deepwalk(model, dataset, config)
    else:
        results = train_gnn_model(model, dataset, config)
    
    # Print results
    print("\nResults:")
    if isinstance(results, dict) and 'metrics' in results:
        metrics = results['metrics']
    else:
        metrics = results
        
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and results
    if config.logging.save_checkpoints:
        os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(config.logging.checkpoint_dir, 
                                 f"{config.model.model_type}_{config.data.dataset_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Save results
        results_path = os.path.join(config.logging.checkpoint_dir,
                                  f"{config.model.model_type}_{config.data.dataset_name}_results.json")
        import json
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Finish wandb run
    if args.wandb and config.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
