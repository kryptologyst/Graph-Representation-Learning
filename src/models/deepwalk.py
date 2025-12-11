"""DeepWalk implementation for graph representation learning."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class DeepWalk:
    """DeepWalk implementation for unsupervised graph representation learning."""
    
    def __init__(self, embedding_dim: int = 64, window_size: int = 5, 
                 num_walks: int = 10, walk_length: int = 20, 
                 workers: int = 4, epochs: int = 10, seed: int = 42):
        """
        Initialize DeepWalk model.
        
        Args:
            embedding_dim: Dimension of node embeddings.
            window_size: Context window size for Skip-gram.
            num_walks: Number of random walks per node.
            walk_length: Length of each random walk.
            workers: Number of workers for Word2Vec training.
            epochs: Number of training epochs.
            seed: Random seed.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        
        self.model: Optional[Word2Vec] = None
        self.node_embeddings: Optional[Dict[int, np.ndarray]] = None
        
    def fit(self, graph: nx.Graph) -> None:
        """
        Train DeepWalk model on the graph.
        
        Args:
            graph: NetworkX graph.
        """
        # Generate random walks
        walks = self._generate_random_walks(graph)
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )
        
        # Extract node embeddings
        self.node_embeddings = {
            int(node): self.model.wv[node] 
            for node in self.model.wv.index_to_key
        }
        
    def _generate_random_walks(self, graph: nx.Graph) -> List[List[str]]:
        """Generate random walks on the graph."""
        random.seed(self.seed)
        walks = []
        nodes = list(graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                current_node = node
                
                while len(walk) < self.walk_length:
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break
                        
                    next_node = random.choice(neighbors)
                    walk.append(str(next_node))
                    current_node = next_node
                    
                walks.append(walk)
                
        return walks
        
    def get_embeddings(self) -> Dict[int, np.ndarray]:
        """
        Get learned node embeddings.
        
        Returns:
            Dictionary mapping node IDs to embeddings.
        """
        if self.node_embeddings is None:
            raise ValueError("Model not trained yet")
        return self.node_embeddings
        
    def get_embedding_matrix(self) -> np.ndarray:
        """
        Get embedding matrix for all nodes.
        
        Returns:
            Embedding matrix of shape (num_nodes, embedding_dim).
        """
        if self.node_embeddings is None:
            raise ValueError("Model not trained yet")
            
        nodes = sorted(self.node_embeddings.keys())
        embeddings = np.array([self.node_embeddings[node] for node in nodes])
        return embeddings
        
    def evaluate_node_classification(self, graph: nx.Graph, labels: Dict[int, int],
                                   test_size: float = 0.3) -> Dict[str, float]:
        """
        Evaluate embeddings on node classification task.
        
        Args:
            graph: NetworkX graph.
            labels: Node labels dictionary.
            test_size: Fraction of data to use for testing.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        if self.node_embeddings is None:
            raise ValueError("Model not trained yet")
            
        # Prepare data
        nodes = list(self.node_embeddings.keys())
        X = np.array([self.node_embeddings[node] for node in nodes])
        y = np.array([labels.get(node, 0) for node in nodes])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        # Train classifier
        clf = SVC(kernel='rbf', random_state=self.seed)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.decision_function(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        
        # AUC (handle binary case)
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc = 0.0
            
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'auc': auc
        }
        
    def get_tsne_embeddings(self, perplexity: float = 30.0, 
                           n_iter: int = 1000) -> np.ndarray:
        """
        Get t-SNE projection of embeddings.
        
        Args:
            perplexity: t-SNE perplexity parameter.
            n_iter: Number of iterations.
            
        Returns:
            t-SNE projected embeddings.
        """
        if self.node_embeddings is None:
            raise ValueError("Model not trained yet")
            
        embeddings = self.get_embedding_matrix()
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=self.seed)
        return tsne.fit_transform(embeddings)
