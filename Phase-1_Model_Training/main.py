"""
Main script for Twitter sentiment analysis model training.

This script orchestrates the DistilBERT fine-tuning process for sentiment analysis.
"""
import argparse
import sys
from pathlib import Path
import torch
import mlflow
import gc
from datetime import datetime

# Add the project root to Python's module search path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logging_utils import get_logger
from src.data.data_loader import DataManager
from src.model.model_manager import ModelManager
from src.training.trainer import Trainer
from src.config.constants import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, LOGS_DIR
from src.config.config import DEFAULT_MODEL_CONFIG

# Set up logger
logger = get_logger(__name__, LOGS_DIR / "main.log")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for Twitter sentiment analysis")
    
    parser.add_argument("--full-finetune", action="store_true", 
                       help="Fine-tune the entire model (default: partial fine-tuning of only last layer)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MODEL_CONFIG['batch_size'],
                       help=f"Batch size (default: {DEFAULT_MODEL_CONFIG['batch_size']})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_MODEL_CONFIG['epochs'],
                       help=f"Number of epochs (default: {DEFAULT_MODEL_CONFIG['epochs']})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_MODEL_CONFIG['learning_rate'],
                       help=f"Learning rate (default: {DEFAULT_MODEL_CONFIG['learning_rate']})")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MODEL_CONFIG['max_len'],
                       help=f"Maximum sequence length (default: {DEFAULT_MODEL_CONFIG['max_len']})")
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL_CONFIG['seed'],
                       help=f"Random seed (default: {DEFAULT_MODEL_CONFIG['seed']})")
    parser.add_argument("--data-portion", type=int, default=DEFAULT_MODEL_CONFIG['portion'],
                       help=f"Data portion denominator (1/portion) (default: {DEFAULT_MODEL_CONFIG['portion']})")
    
    return parser.parse_args()

def main():
    """Run the complete model training process."""
    args = parse_args()
    
    # Set configuration
    config = DEFAULT_MODEL_CONFIG.copy()
    config.update({
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'max_len': args.max_len,
        'seed': args.seed,
        'portion': args.data_portion,
        'partial_freeze': not args.full_finetune
    })
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"distilbert_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Starting Twitter sentiment analysis model training")
            logger.info(f"Configuration: {config}")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize model manager
            model_manager = ModelManager(config)
            model, tokenizer = model_manager.setup_model_and_tokenizer()
            
            # Load and prepare data
            data_manager = DataManager(config)
            train_loader, val_loader = data_manager.prepare_data(tokenizer)
            
                        # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            
            # Create optimizer and scheduler
            optimizer, scheduler = model_manager.get_optimizer_and_scheduler(len(train_loader))
            
            # Initialize trainer
            trainer = Trainer(model, config, device)
            
            # Train model
            logger.info("Starting training...")
            metrics = trainer.train(train_loader, val_loader, optimizer, scheduler)
            
            # Log final results
            logger.info("\nTraining completed successfully!")
            logger.info(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
            logger.info(f"Total training time: {metrics['total_training_time']:.2f} seconds")
            
            # Clean up
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("Clean up completed")
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()