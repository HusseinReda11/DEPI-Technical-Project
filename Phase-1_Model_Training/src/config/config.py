"""
Configuration settings for the model training module.
"""
from typing import Dict, Any, List, Optional, Tuple

# Default model configuration
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    'max_len': 48,
    'batch_size': 16,
    'portion': 1000,  # Use 1/portion of the data
    'epochs': 3,
    'learning_rate': 2e-5,
    'partial_freeze': True,  # By default, freeze most of the model
    'num_warmup_steps_ratio': 0.1,
    'weight_decay': 0.01,
    'seed': 42,
    'model_name': 'distilbert-base-uncased',
    'num_labels': 2
}

# Visualization configuration
FIGURE_SIZE: Tuple[int, int] = (10, 8)
FIGURE_DPI: int = 300