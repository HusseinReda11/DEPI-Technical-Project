"""
Sentiment analysis model implementation using DistilBERT.
"""
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Union

# Import from parent directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODELS_DIR, MODEL_FILE, TOKENIZER_NAME, MAX_LEN, NUM_LABELS

# Set up logging
logger = logging.getLogger(__name__)

class SentimentModel:
    """
    DistilBERT-based sentiment analysis model for Twitter sentiment classification.
    
    This class handles loading the model, preprocessing inputs, and making predictions.
    """
    
    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Initialize the sentiment analysis model.
        
        Args:
            device: PyTorch device to use for inference (defaults to CPU)
        """
        self.device = device if device is not None else torch.device("cpu")
        self.model = None
        self.tokenizer = None
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the tokenizer and model from the checkpoint."""
        try:
            logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
            
            logger.info(f"Loading base model: {TOKENIZER_NAME}")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                TOKENIZER_NAME, 
                num_labels=NUM_LABELS
            )
            
            model_path = MODELS_DIR / MODEL_FILE
            
            if model_path.exists():
                logger.info(f"Loading checkpoint from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Try different ways to load the model depending on the checkpoint format
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                logger.info("Model checkpoint loaded successfully")
            else:
                logger.warning(f"No checkpoint found at {model_path}. Using pretrained model.")
                
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, tweets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a list of tweets.
        
        Args:
            tweets: List of dictionaries, each containing a 'text' key with the tweet content
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not initialized")
            raise RuntimeError("Model or tokenizer not initialized")
        
        for tweet_data in tweets:
            # Extract and validate text
            tweet = tweet_data.get('text', '')
            if not tweet or not isinstance(tweet, str):
                results.append({
                    'error': 'Invalid or missing text field',
                    'text': tweet
                })
                continue
                
            # Prepare input
            try:
                inputs = self.tokenizer(
                    tweet, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LEN
                )
                
                # Move to device
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                
                # Perform inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process outputs
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()
                
                # Create result
                sentiment = "positive" if pred_class == 1 else "negative"
                results.append({
                    'text': tweet,
                    'sentiment': sentiment,
                    'confidence': confidence
                })
                
            except Exception as e:
                logger.error(f"Error predicting sentiment for tweet: {e}", exc_info=True)
                results.append({
                    'text': tweet,
                    'error': f"Prediction error: {str(e)}"
                })
        
        return results
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text string.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with prediction results
        """
        results = self.predict([{'text': text}])
        return results[0]