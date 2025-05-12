"""
Global configuration settings for the sentiment analysis application.
"""
import os
from pathlib import Path

# Application configuration
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 8080))

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Model configuration
MODEL_FILE = "destillbert.pt"
TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LEN = 128
NUM_LABELS = 2