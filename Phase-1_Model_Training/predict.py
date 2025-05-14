"""
Script for making predictions with a trained DistilBERT sentiment analysis model.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.model_inference import SentimentPredictor
from src.utils.logging_utils import get_logger
from src.config.constants import MODEL_DIR, PREDICTIONS_LOGS_DIR

logger = get_logger(__name__, log_file=PREDICTIONS_LOGS_DIR / "predictions.log")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict sentiment with trained DistilBERT model")
    
    parser.add_argument("--model-path", type=str, 
                       help="Path to the model checkpoint (default: latest in models directory)")
    parser.add_argument("--text", type=str, 
                       help="Text to predict sentiment for")
    parser.add_argument("--input-file", type=str, 
                       help="CSV file with texts to predict")
    parser.add_argument("--text-column", type=str, default="text", 
                       help="Name of the text column in the input file")
    parser.add_argument("--output-file", type=str, 
                       help="CSV file to save predictions to")
    parser.add_argument("--with-probs", action="store_true", 
                       help="Include prediction probabilities in output")
    
    return parser.parse_args()

def find_latest_model() -> Optional[Path]:
    """
    Find the latest model checkpoint in the models directory.
    
    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    checkpoints = list(MODEL_DIR.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by timestamp in filename
    latest = sorted(checkpoints, key=lambda p: p.stem.split('_')[-1], reverse=True)[0]
    return latest

def main():
    """Run the prediction script."""
    args = parse_args()
    
    # Get model path
    model_path = args.model_path
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            logger.error("No model checkpoints found. Please specify model path.")
            return
    else:
        model_path = Path(model_path)
    
    logger.info(f"Using model from {model_path}")
    
    # Initialize predictor
    predictor = SentimentPredictor(model_path)
    
    # Make predictions
    if args.text:
        # Single text prediction
        result = predictor.predict(args.text, return_probabilities=args.with_probs)
        
        if args.with_probs:
            pred, prob = result[0]
            sentiment = "Positive" if pred == 1 else "Negative"
            logger.info(f"Text: {args.text}")
            logger.info(f"Sentiment: {sentiment} (confidence: {prob:.4f})")
        else:
            sentiment = "Positive" if result[0] == 1 else "Negative"
            logger.info(f"Text: {args.text}")
            logger.info(f"Sentiment: {sentiment}")
    
    elif args.input_file:
        # Batch prediction
        input_file = Path(args.input_file)
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
            
        # Load input data
        try:
            df = pd.read_csv(input_file)
            if args.text_column not in df.columns:
                logger.error(f"Text column '{args.text_column}' not found in input file")
                return
                
            texts = df[args.text_column].tolist()
            
        except Exception as e:
            logger.error(f"Error loading input file: {e}")
            return
            
        # Make predictions
        results = predictor.predict(texts, return_probabilities=args.with_probs)
        
        # Process results
        if args.with_probs:
            preds, probs = zip(*results)
            df['sentiment_prediction'] = preds
            df['sentiment_probability'] = probs
            df['sentiment_label'] = df['sentiment_prediction'].apply(lambda x: "Positive" if x == 1 else "Negative")
        else:
            df['sentiment_prediction'] = results
            df['sentiment_label'] = df['sentiment_prediction'].apply(lambda x: "Positive" if x == 1 else "Negative")
        
        # Save results
        output_file = args.output_file or input_file.with_name(f"{input_file.stem}_predictions.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    
    else:
        logger.error("Please specify either --text or --input-file")

if __name__ == "__main__":
    main()