"""
Visualization utilities for model training results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from ..config.constants import FIGURES_DIR
from ..config.config import FIGURE_SIZE, FIGURE_DPI
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelVisualizer:
    """Class for visualizing model training results."""
    
    def __init__(self, save_dir: Path = FIGURES_DIR) -> None:
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                              labels: List[str] = ["Negative", "Positive"]) -> Path:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels for the confusion matrix
            
        Returns:
            Path to saved figure
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=FIGURE_SIZE)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        save_path = self.save_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
        return save_path
    
    def plot_roc_curve(self, y_true: List[int], y_prob: List[float]) -> Path:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            
        Returns:
            Path to saved figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        save_path = self.save_dir / "roc_curve.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}")
        return save_path
    
    def plot_precision_recall_curve(self, y_true: List[int], y_prob: List[float]) -> Path:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for the positive class
            
        Returns:
            Path to saved figure
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        save_path = self.save_dir / "precision_recall_curve.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-recall curve saved to {save_path}")
        return save_path
    
    def plot_learning_curves(self, train_losses: List[float], val_losses: List[float], 
                           accuracies: List[float]) -> Path:
        """
        Plot learning curves.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            accuracies: Validation accuracies per epoch
            
        Returns:
            Path to saved figure
        """
        epochs = range(1, len(train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0]*2, FIGURE_SIZE[1]))
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curve
        ax2.plot(epochs, accuracies, 'g-', label='Validation Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        save_path = self.save_dir / "learning_curves.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Learning curves saved to {save_path}")
        return save_path
    
    def plot_resource_usage(self, resource_data: List[Dict[str, float]]) -> Path:
        """
        Plot resource usage during training.
        
        Args:
            resource_data: List of dictionaries containing resource usage data
            
        Returns:
            Path to saved figure
        """
        df = pd.DataFrame(resource_data)
        
        plt.figure(figsize=FIGURE_SIZE)
        plt.plot(df['cpu'], label='CPU Usage (%)')
        plt.plot(df['ram'], label='RAM Usage (%)')
        if 'gpu_util' in df.columns and df['gpu_util'].mean() > 0:
            plt.plot(df['gpu_util'], label='GPU Utilization (%)')
            plt.plot(df['gpu_memory'], label='GPU Memory Usage (%)')
        
        plt.title('Resource Usage During Training')
        plt.xlabel('Measurement Points')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = self.save_dir / "resource_usage.png"
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Resource usage plot saved to {save_path}")
        return save_path