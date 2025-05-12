"""
Inference utilities for the trained DistilBERT model.
"""
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..config.constants import MODEL_DIR

logger = get_logger(__name__)

class SentimentPredictor:
    """Class for making sentiment predictions with a trained DistilBERT model."""
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[torch.device] = None) -> None:
        """
        Initialize the sentiment predictor.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to use for inference
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.config = None
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: Path) -> None:
        """
        Load a trained model from a checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
        """
        logger.info(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Load tokenizer
        model_name = self.config.get('model_name', 'distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.config.get('num_labels', 2)
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Union[List[int], List[Tuple[int, float]]]:
        """
        Make sentiment predictions on text(s).
        
        Args:
            texts: Single text or list of texts to predict
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            If return_probabilities is False, returns a list of predictions (0 for negative, 1 for positive)
            If return_probabilities is True, returns a list of tuples (prediction, probability)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before making predictions")
            
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        logger.info(f"Making predictions on {len(texts)} texts")
        
        # Tokenize
        max_len = self.config.get('max_len', 48)
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
        # Process outputs
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        # Prepare results
        result = []
        if return_probabilities:
            for pred, prob in zip(predictions.cpu().numpy(), probabilities.cpu().numpy()):
                result.append((int(pred), float(prob[pred])))
        else:
            result = predictions.cpu().numpy().tolist()
        
        return result