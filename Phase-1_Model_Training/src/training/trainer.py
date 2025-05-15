"""
Training utilities for fine-tuning DistilBERT.
"""
import torch
import numpy as np
import time
from datetime import datetime
import mlflow
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
import os
import psutil
from pathlib import Path
import GPUtil
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging_utils import get_logger
from ..config.constants import FIGURES_DIR, MODEL_DIR
from ..config.config import FIGURE_SIZE, FIGURE_DPI

logger = get_logger(__name__)

class MetricsTracker:
    """Class for tracking training metrics."""
    
    def __init__(self) -> None:
        """Initialize the metrics tracker."""
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.epoch_times: List[float] = []
        self.training_times: List[float] = []
        self.evaluation_times: List[float] = []
        self.training_samples: int = 0
        self.evaluation_samples: int = 0
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.accuracies: List[float] = []
        self.resource_usage: List[Dict[str, float]] = []
        self.predictions: List[int] = []
        self.true_labels: List[int] = []
        
    def start_training(self) -> None:
        """Mark the start of training."""
        self.start_time = time.time()
        
    def start_epoch(self) -> None:
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        
    def end_epoch(self, train_loss: float, val_loss: float) -> None:
        """
        Mark the end of an epoch and record metrics.
        
        Args:
            train_loss: Training loss for the epoch
            val_loss: Validation loss for the epoch
        """
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def add_batch_training_time(self, batch_time: float, batch_size: int) -> None:
        """
        Record batch training time.
        
        Args:
            batch_time: Time taken for the batch
            batch_size: Number of samples in the batch
        """
        self.training_times.append(batch_time)
        self.training_samples += batch_size
        
    def add_batch_evaluation_time(self, batch_time: float, batch_size: int) -> None:
        """
        Record batch evaluation time.
        
        Args:
            batch_time: Time taken for the batch
            batch_size: Number of samples in the batch
        """
        self.evaluation_times.append(batch_time)
        self.evaluation_samples += batch_size
        
    def add_predictions(self, preds: List[int], labels: List[int]) -> None:
        """
        Record predictions and true labels.
        
        Args:
            preds: Model predictions
            labels: True labels
        """
        self.predictions.extend(preds)
        self.true_labels.extend(labels)
        
    def add_accuracy(self, accuracy: float) -> None:
        """
        Record accuracy.
        
        Args:
            accuracy: Validation accuracy
        """
        self.accuracies.append(accuracy)
        
    def add_resource_usage(self, cpu_percent: float, ram_percent: float, 
                          gpu_percent: float, gpu_memory: float) -> None:
        """
        Record system resource usage.
        
        Args:
            cpu_percent: CPU usage percentage
            ram_percent: RAM usage percentage
            gpu_percent: GPU utilization percentage
            gpu_memory: GPU memory usage percentage
        """
        self.resource_usage.append({
            'cpu': cpu_percent,
            'ram': ram_percent,
            'gpu_util': gpu_percent,
            'gpu_memory': gpu_memory
        })
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        total_training_time = time.time() - self.start_time
        avg_training_time = sum(self.training_times) / self.training_samples if self.training_samples > 0 else 0
        avg_evaluation_time = sum(self.evaluation_times) / self.evaluation_samples if self.evaluation_samples > 0 else 0
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
        
        # Calculate average resource usage
        avg_resources = {
            'cpu': np.mean([r['cpu'] for r in self.resource_usage]) if self.resource_usage else 0,
            'ram': np.mean([r['ram'] for r in self.resource_usage]) if self.resource_usage else 0,
            'gpu_util': np.mean([r['gpu_util'] for r in self.resource_usage]) if self.resource_usage else 0,
            'gpu_memory': np.mean([r['gpu_memory'] for r in self.resource_usage]) if self.resource_usage else 0
        }
        
        return {
            'avg_training_time_per_sample': avg_training_time,
            'avg_evaluation_time_per_sample': avg_evaluation_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'first_epoch_accuracy': self.accuracies[0] if self.accuracies else 0,
            'final_accuracy': self.accuracies[-1] if self.accuracies else 0,
            'avg_epoch_time': avg_epoch_time,
            'total_training_time': total_training_time,
            'avg_resources': avg_resources,
            'predictions': self.predictions,
            'true_labels': self.true_labels
        }
        
    def plot_confusion_matrix(self, save_path: Path = FIGURES_DIR / "confusion_matrix.png") -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            save_path: Path to save the confusion matrix image
        """
        if not self.predictions or not self.true_labels:
            logger.warning("No predictions available to create confusion matrix")
            return
            
        cm = confusion_matrix(self.true_labels, self.predictions)
        plt.figure(figsize=FIGURE_SIZE)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
        
    def plot_learning_curves(self, save_path: Path = FIGURES_DIR / "learning_curves.png") -> None:
        """
        Plot and save learning curves.
        
        Args:
            save_path: Path to save the learning curves image
        """
        if not self.train_losses or not self.val_losses:
            logger.warning("No loss data available to plot learning curves")
            return
            
        plt.figure(figsize=FIGURE_SIZE)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning curves saved to {save_path}")


    def get_system_resources() -> Dict[str, float]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary containing CPU, RAM, GPU utilization and memory usage
        """
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        
        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
            else:
                gpu_util = 0
                gpu_memory = 0
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            gpu_util = 0
            gpu_memory = 0
        
        return {
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'gpu_util': gpu_util,
            'gpu_memory': gpu_memory
        }


class Trainer:
    """Class for training DistilBERT model."""
    
    def __init__(self, model: DistilBertForSequenceClassification, config: Dict[str, Any], 
                device: torch.device) -> None:
        """
        Initialize the trainer.
        
        Args:
            model: DistilBERT model
            config: Training configuration
            device: Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device
        self.metrics = MetricsTracker()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
            optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary containing final metrics
        """
        try:
            # Start tracking metrics
            self.metrics.start_training()
            
            # Initialize MLflow tracking
            mlflow.log_params(self.config)
            mlflow.log_param("model_size_mb", self._get_model_size())
            mlflow.log_param("trainable_params", self._get_trainable_params())
            
            best_val_accuracy = 0.0
            
            # Training loop
            for epoch in range(self.config['epochs']):
                self.metrics.start_epoch()
                logger.info(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
                
                # Training phase
                train_loss = self._run_train_epoch(train_loader, optimizer, scheduler, epoch)
                
                # Validation phase
                val_loss, accuracy = self._run_validation(val_loader, epoch)
                
                # Record metrics
                self.metrics.end_epoch(train_loss, val_loss)
                
                # Log to MLflow
                mlflow.log_metrics({
                    f"train_loss_epoch_{epoch+1}": train_loss,
                    f"val_loss_epoch_{epoch+1}": val_loss,
                    f"accuracy_epoch_{epoch+1}": accuracy
                }, step=epoch)
                
                # Save if best model
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    checkpoint_path = self._save_checkpoint(epoch, val_loss, accuracy)
                    mlflow.log_artifact(checkpoint_path)
            
            # Generate and save visualizations
            self.metrics.plot_confusion_matrix()
            self.metrics.plot_learning_curves()
            
            # Log artifacts to MLflow
            mlflow.log_artifact(str(FIGURES_DIR / "confusion_matrix.png"))
            mlflow.log_artifact(str(FIGURES_DIR / "learning_curves.png"))
            
            # Log classification report
            classification_rep = classification_report(
                self.metrics.true_labels, 
                self.metrics.predictions,
                output_dict=True
            )
            
            # Log classification metrics to MLflow
            mlflow.log_metrics({
                "precision_class_0": classification_rep['0']['precision'],
                "recall_class_0": classification_rep['0']['recall'],
                "f1_score_class_0": classification_rep['0']['f1-score'],
                "precision_class_1": classification_rep['1']['precision'],
                "recall_class_1": classification_rep['1']['recall'],
                "f1_score_class_1": classification_rep['1']['f1-score'],
                "accuracy": classification_rep['accuracy']
            })
            
            # Calculate and log final metrics
            final_metrics = self.metrics.get_metrics()
            
            # Log resource metrics
            mlflow.log_metrics({
                "avg_epoch_time": final_metrics['avg_epoch_time'],
                "total_training_time": final_metrics['total_training_time'],
                "avg_training_time_per_sample_ms": final_metrics['avg_training_time_per_sample'] * 1000,
                "avg_evaluation_time_per_sample_ms": final_metrics['avg_evaluation_time_per_sample'] * 1000,
                "avg_cpu_usage": final_metrics['avg_resources']['cpu'],
                "avg_ram_usage": final_metrics['avg_resources']['ram'],
                "avg_gpu_util": final_metrics['avg_resources']['gpu_util'],
                "avg_gpu_memory": final_metrics['avg_resources']['gpu_memory']
            })
            
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise
    
    def _run_train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int) -> float:
        """
        Run one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
            batch_start = time.time()
            
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward and backward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            outputs.loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record metrics
            batch_time = time.time() - batch_start
            self.metrics.add_batch_training_time(batch_time, len(input_ids))
            running_loss += outputs.loss.item()
            
            # Monitor resources every 10 batches
            if batch_idx % 10 == 0:
                resources = MetricsTracker.get_system_resources()
                self.metrics.add_resource_usage(
                    resources['cpu_percent'],
                    resources['ram_percent'],
                    resources['gpu_util'],
                    resources['gpu_memory']
                )
        
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _run_validation(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Run validation.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Tuple of (average validation loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch_start = time.time()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=targets
                )
                
                batch_time = time.time() - batch_start
                self.metrics.add_batch_evaluation_time(batch_time, len(input_ids))
                
                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                # Store predictions and labels for confusion matrix
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Store predictions for confusion matrix
        self.metrics.add_predictions(all_preds, all_labels)
        self.metrics.add_accuracy(accuracy)
        
        logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
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
    
    def _get_trainable_params(self) -> int:
        """
        Get the number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, accuracy: float) -> str:
        """
        Save a model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            accuracy: Validation accuracy
            
        Returns:
            Path to the saved checkpoint
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch+1}_{timestamp}.pt')
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'accuracy': accuracy,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path