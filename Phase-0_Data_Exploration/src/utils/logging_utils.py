import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.constants import REPORTS_DIR

# Configure logging with both file and console handlers
os.makedirs(REPORTS_DIR, exist_ok=True)
log_file = os.path.join(REPORTS_DIR, "data_exploration.log")

def get_logger(
    name: str,
    log_file: str = log_file,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Get a logger with both file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove old handlers to avoid duplicate logs in interactive environments
    if logger.hasHandlers():
        logger.handlers.clear()

    # Make sure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File handler
    f_handler = logging.FileHandler(log_file, mode='a')
    f_handler.setLevel(level)
    f_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_formatter)

    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(level)
    c_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_formatter)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    return logger