"""Interactive demo for graph representation learning."""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from src.data.dataset import get_dataset
from src.models.deepwalk import DeepWalk
from src.models.dgi import DGI
from src.models.graphcl import GraphCL
from src.models.grace import GRACE
from src.utils.device import set_seed


# Page configuration
st.set_page_config(
    page_title="Graph Representation Learning Demo",
    page_icon="🕸️",
    layout="wide"
)

# Title
st.title("🕸️ Graph Representation Learning Demo")
st.markdown("""
This demo showcases different graph representation learning methods including DeepWalk, 
DGI (Deep Graph Infomax), GraphCL, and GRACE. Explore how these methods learn meaningful 
node embeddings and perform on various tasks.
""")


@st.cache_data
def load_dataset(dataset_name: str) -> Tuple[nx.Graph, torch.Tensor, torch.Tensor]:
    """Load and cache dataset."""
    dataset = get_dataset(dataset_name)
    return dataset.graph, dataset.node_features, dataset.node_labels


@st.cache_data
def train_deepwalk(graph: nx.Graph, embedding_dim: int, num_walks: int, 
                  walk_length: int, window_size: int) -> DeepWalk:
    """Train and cache DeepWalk model."""
    model = DeepWalk(
        embedding_dim=embedding_dim,
        num_walks=num_walks,
        walk_length=walk_length,
        window_size=window_size
    )
    model.fit(graph)
    return model


@st.cache_data
def train_gnn_model(model_type: str, data, config: Dict) -> torch.nn.Module:
    """Train and cache GNN model."""
    input_dim = data.x.shape[1]
    
    if model_type == "dgi":
        model = DGI(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif model_type == "graphcl":
        model = GraphCL(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    elif model_type == "grace":
        model = GRACE(
            input_dim=input_dim,
            hidden_dim=config['hidden_dim'],
            output_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        
        if hasattr(model, 'contrastive_loss'):
            loss = model.contrastive_loss(data.x, data.edge_index)
        elif hasattr(model, 'loss'):
            x_corrupted = model.corrupt_features(data.x)
            loss = model.loss(data.x, data.edge_index, x_corrupted)
        else:
            loss = torch.tensor(0.0)
            
        loss.backward()
        optimizer.step()
    
    return model


def visualize_graph(graph: nx.Graph, labels: Optional[torch.Tensor] = None) -> go.Figure:
    """Create interactive graph visualization."""
    pos = nx.spring_layout(graph, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'Node {node}')
        
        if labels is not None:
            node_colors.append(labels[node].item())
        else:
            node_colors.append(0)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f'Node {i}' for i in range(len(node_x))],
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_colors,
            size=20,
            colorbar=dict(
                thickness=15,
                title="Node Class",
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2, color='black')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Graph Visualization',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='black', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig


def visualize_embeddings(embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
                        title: str = "Embeddings Visualization") -> go.Figure:
    """Create interactive embeddings visualization."""
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'node': [f'Node {i}' for i in range(len(embeddings_2d))]
    })
    
    if labels is not None:
        df['class'] = labels
        fig = px.scatter(df, x='x', y='y', color='class', 
                        hover_data=['node'], title=title)
    else:
        fig = px.scatter(df, x='x', y='y', hover_data=['node'], title=title)
    
    fig.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        width=600,
        height=500
    )
    
    return fig


def evaluate_embeddings(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate embeddings on node classification."""
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train classifier
    clf = SVC(kernel='rbf', random_state=42)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


def main():
    """Main demo function."""
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["karate", "cora", "synthetic"],
        help="Choose the dataset to work with"
    )
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model",
        ["deepwalk", "dgi", "graphcl", "grace"],
        help="Choose the graph representation learning method"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    embedding_dim = st.sidebar.slider("Embedding Dimension", 16, 128, 64)
    
    if model_type == "deepwalk":
        num_walks = st.sidebar.slider("Number of Walks", 5, 50, 10)
        walk_length = st.sidebar.slider("Walk Length", 10, 50, 20)
        window_size = st.sidebar.slider("Window Size", 3, 10, 5)
    else:
        hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 128, 64)
        num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.1)
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001)
        epochs = st.sidebar.slider("Epochs", 10, 200, 50)
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        graph, node_features, node_labels = load_dataset(dataset_name)
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", graph.number_of_nodes())
    with col2:
        st.metric("Edges", graph.number_of_edges())
    with col3:
        st.metric("Features", node_features.shape[1] if node_features is not None else "None")
    with col4:
        st.metric("Classes", len(torch.unique(node_labels)) if node_labels is not None else "None")
    
    # Train model
    if st.button("Train Model", type="primary"):
        with st.spinner(f"Training {model_type}..."):
            if model_type == "deepwalk":
                model = train_deepwalk(graph, embedding_dim, num_walks, walk_length, window_size)
                embeddings = model.get_embedding_matrix()
            else:
                # Prepare data for GNN models
                from torch_geometric.data import Data
                edge_list = list(graph.edges())
                if edge_list:
                    edge_index = torch.tensor(edge_list).t().contiguous()
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                
                data = Data(x=node_features, edge_index=edge_index)
                
                config = {
                    'hidden_dim': hidden_dim,
                    'embedding_dim': embedding_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'learning_rate': learning_rate,
                    'epochs': epochs
                }
                
                model = train_gnn_model(model_type, data, config)
                
                # Get embeddings
                model.eval()
                with torch.no_grad():
                    embeddings = model(data.x, data.edge_index).cpu().numpy()
        
        # Store results in session state
        st.session_state['model'] = model
        st.session_state['embeddings'] = embeddings
        st.session_state['model_type'] = model_type
        
        st.success(f"{model_type.upper()} training completed!")
    
    # Display results if model is trained
    if 'embeddings' in st.session_state:
        st.header("Results")
        
        embeddings = st.session_state['embeddings']
        model_type = st.session_state['model_type']
        
        # Evaluation metrics
        if node_labels is not None:
            labels_np = node_labels.cpu().numpy()
            metrics = evaluate_embeddings(embeddings, labels_np)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("F1 Macro", f"{metrics['f1_macro']:.3f}")
            with col3:
                st.metric("F1 Micro", f"{metrics['f1_micro']:.3f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Graph Structure")
            fig_graph = visualize_graph(graph, node_labels)
            st.plotly_chart(fig_graph, use_container_width=True)
        
        with col2:
            st.subheader("Learned Embeddings")
            fig_emb = visualize_embeddings(
                embeddings, 
                node_labels.cpu().numpy() if node_labels is not None else None,
                f"{model_type.upper()} Embeddings"
            )
            st.plotly_chart(fig_emb, use_container_width=True)
        
        # Embedding analysis
        st.subheader("Embedding Analysis")
        
        # Similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        fig_sim = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='Viridis',
            title="Node Similarity Matrix"
        ))
        fig_sim.update_layout(width=500, height=500)
        st.plotly_chart(fig_sim)
        
        # Embedding statistics
        st.subheader("Embedding Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Embedding Statistics:**")
            st.write(f"- Mean: {np.mean(embeddings):.4f}")
            st.write(f"- Std: {np.std(embeddings):.4f}")
            st.write(f"- Min: {np.min(embeddings):.4f}")
            st.write(f"- Max: {np.max(embeddings):.4f}")
        
        with col2:
            st.write("**Dimension-wise Statistics:**")
            dim_stats = pd.DataFrame({
                'Dimension': range(embeddings.shape[1]),
                'Mean': np.mean(embeddings, axis=0),
                'Std': np.std(embeddings, axis=0)
            })
            st.dataframe(dim_stats, use_container_width=True)


if __name__ == "__main__":
    main()
