"""
Models package for sentiment analysis application.
"""
from .sentiment_model import SentimentModel

# Initialize model as a singleton to avoid loading multiple times
_model_instance = None

def get_model():
    """Get or create the singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance