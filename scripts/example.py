"""Example script demonstrating graph representation learning."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import get_dataset
from src.models.deepwalk import DeepWalk
from src.models.dgi import DGI
from src.eval.evaluator import Evaluator
from src.utils.device import set_seed


def main():
    """Main example function."""
    print("Graph Representation Learning Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load dataset
    print("\n1. Loading Karate Club dataset...")
    dataset = get_dataset("karate")
    
    print(f"   Nodes: {dataset.graph.number_of_nodes()}")
    print(f"   Edges: {dataset.graph.number_of_edges()}")
    print(f"   Features: {dataset.node_features.shape[1] if dataset.node_features is not None else 'None'}")
    print(f"   Classes: {len(torch.unique(dataset.node_labels)) if dataset.node_labels is not None else 'None'}")
    
    # Train DeepWalk
    print("\n2. Training DeepWalk...")
    deepwalk_model = DeepWalk(
        embedding_dim=32,
        num_walks=5,
        walk_length=10,
        window_size=3
    )
    deepwalk_model.fit(dataset.graph)
    
    # Evaluate DeepWalk
    print("\n3. Evaluating DeepWalk...")
    deepwalk_metrics = deepwalk_model.evaluate_node_classification(
        dataset.graph,
        {i: dataset.node_labels[i].item() for i in range(dataset.node_labels.size(0))}
    )
    
    print("   DeepWalk Results:")
    for metric, value in deepwalk_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Train DGI
    print("\n4. Training DGI...")
    data = dataset.get_pyg_data()
    dgi_model = DGI(
        input_dim=data.x.shape[1],
        hidden_dim=32,
        output_dim=32,
        num_layers=2
    )
    
    # Simple training loop
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.01)
    
    print("   Training DGI for 20 epochs...")
    for epoch in range(20):
        optimizer.zero_grad()
        x_corrupted = dgi_model.corrupt_features(data.x)
        loss = dgi_model.loss(data.x, data.edge_index, x_corrupted)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"     Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    # Evaluate DGI
    print("\n5. Evaluating DGI...")
    evaluator = Evaluator()
    dgi_metrics = evaluator.evaluate_node_classification(
        dgi_model, data, data.train_mask, data.val_mask, data.test_mask
    )
    
    print("   DGI Results:")
    for metric, value in dgi_metrics.items():
        print(f"     {metric}: {value:.4f}")
    
    # Compare results
    print("\n6. Model Comparison:")
    print("   " + "-" * 40)
    print("   Model      Accuracy    F1-Macro    F1-Micro")
    print("   " + "-" * 40)
    print(f"   DeepWalk   {deepwalk_metrics['accuracy']:.4f}      {deepwalk_metrics['f1_macro']:.4f}      {deepwalk_metrics['f1_micro']:.4f}")
    print(f"   DGI        {dgi_metrics['accuracy']:.4f}      {dgi_metrics['f1_macro']:.4f}      {dgi_metrics['f1_micro']:.4f}")
    print("   " + "-" * 40)
    
    # Visualize embeddings
    print("\n7. Generating visualizations...")
    try:
        # DeepWalk visualization
        deepwalk_embeddings = deepwalk_model.get_tsne_embeddings()
        
        # DGI visualization
        dgi_model.eval()
        with torch.no_grad():
            dgi_embeddings = dgi_model(data.x, data.edge_index).cpu().numpy()
        
        evaluator.visualize_embeddings(
            dgi_embeddings, 
            data.y, 
            save_path="assets/plots/dgi_embeddings.png"
        )
        
        print("   Visualizations saved to assets/plots/")
        
    except Exception as e:
        print(f"   Visualization failed: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")


if __name__ == "__main__":
    import torch
    main()
