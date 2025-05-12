"""
Logging utilities for the model training module.
"""
import logging
from typing import Optional
from pathlib import Path
import sys
import os

from ..config.constants import LOGS_DIR

def get_logger(
    name: str, 
    log_file: Optional[Path] = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get a logger with specified configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Check if logger already has handlers to avoid duplicates
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOGS_DIR / "model_training.log"
        
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger