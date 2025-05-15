"""
Model management utilities for fine-tuning DistilBERT.
"""
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Any, Tuple
import os
from datetime import datetime

from ..utils.logging_utils import get_logger
from ..config.constants import MODEL_DIR

logger = get_logger(__name__)

class ModelManager:
    """Class for managing the DistilBERT model lifecycle."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the model manager.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizer]:
        """
        Setup the DistilBERT model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Setting up model and tokenizer from {self.config['model_name']}")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.config['model_name'])
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=self.config['num_labels']
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Freeze layers if partial_freeze is enabled
        if self.config['partial_freeze']:
            self._partial_freeze_model()
            
        logger.info(f"Model size: {self._get_model_size():.2f} MB")
        
        return self.model, self.tokenizer
    
    def _partial_freeze_model(self) -> None:
        """Freeze all layers except the classifier and the last transformer layer."""
        logger.info("Partially freezing model (keeping only classifier and last transformer layer trainable)")
        
        # Freeze embeddings
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False
            
        # Freeze all transformer layers except the last one
        for i, layer in enumerate(self.model.distilbert.transformer.layer):
            # Only unfreeze the last layer (index 5 for DistilBERT)
            if i < 5:  # DistilBERT has 6 layers (0-5)
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Count trainable parameters for logging
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    def _get_model_size(self) -> float:
        """
        Calculate the model size in MB.
        
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / (1024**2)
        return size_all_mb
    
    def save_checkpoint(self, epoch: int, loss: float, accuracy: float, checkpoint_dir: str = str(MODEL_DIR)) -> str:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Validation loss
            accuracy: Validation accuracy
            checkpoint_dir: Directory to save checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def get_optimizer_and_scheduler(self, train_dataloader_length: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Create optimizer and learning rate scheduler.
        
        Args:
            train_dataloader_length: Number of batches in the training dataloader
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = train_dataloader_length * self.config['epochs']
        warmup_steps = int(total_steps * self.config['num_warmup_steps_ratio'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler