"""Configuration management for graph representation learning project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    dataset_name: str = "karate"
    data_dir: str = "data"
    num_walks: int = 10
    walk_length: int = 20
    window_size: int = 5
    embedding_dim: int = 64
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    model_type: str = "deepwalk"  # deepwalk, dgi, graphcl, grace
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    use_batch_norm: bool = True
    use_residual: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = False
    gradient_clip_norm: Optional[float] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "auc"])
    save_embeddings: bool = True
    visualize_embeddings: bool = True
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    use_wandb: bool = False
    wandb_project: str = "graph-representation-learning"
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Main configuration class."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> Config:
        """Load configuration from YAML file."""
        cfg = OmegaConf.load(config_path)
        return cls(
            data=DataConfig(**cfg.data),
            model=ModelConfig(**cfg.model),
            training=TrainingConfig(**cfg.training),
            evaluation=EvaluationConfig(**cfg.evaluation),
            logging=LoggingConfig(**cfg.logging),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.structured(self)
    
    def save_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        OmegaConf.save(self.to_dict(), config_path)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path is None:
        return get_default_config()
    return Config.from_yaml(config_path)
