"""Configuration settings for CNN image classification project."""

import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataConfig:
    """Data configuration."""
    mnist_path: str = "./data/mnist"
    cifar10_path: str = "./data/cifar10"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class TrainingConfig:
    """Training configuration."""
    device: str = "cuda"  # or "cpu"
    seed: int = 42
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adam"  # "adam" or "sgd"
    scheduler: str = "cosine"  # "cosine", "step", or "none"
    early_stopping_patience: int = 10
    
@dataclass
class ModelConfig:
    """Model configuration."""
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
    use_dropout: bool = True

# Default configurations
DATA_CONFIG = DataConfig()
TRAINING_CONFIG = TrainingConfig()
MODEL_CONFIG = ModelConfig()

# Create necessary directories
for path in [DATA_CONFIG.mnist_path, DATA_CONFIG.cifar10_path, "./saved_models", "./results"]:
    os.makedirs(path, exist_ok=True)
