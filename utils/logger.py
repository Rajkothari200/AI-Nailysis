"""
AI Nailysis V2 - Structured Logging Engine
"""

import logging
import sys
from typing import Optional


def get_logger(name: str = "AINailysisV2", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a structured logger instance with formatted console output.
    
    Args:
        name: Name of the logger module.
        log_file: Optional filepath to save log messages to disk.
        level: Logging verbosity level.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
