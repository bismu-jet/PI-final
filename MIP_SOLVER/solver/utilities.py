import logging
import sys

def setup_logger():
    """
    Sets up a centralized logger.

    This function is designed to be idempotent, meaning it can be called
    multiple times without creating duplicate handlers.
    """
    # Get the root logger. All other loggers will inherit from this.
    logger = logging.getLogger()
    
    # Check if handlers have already been added to this logger.
    # If they have, we don't need to do anything.
    if logger.hasHandlers():
        return logger

    # If no handlers exist, configure the logger for the first time.
    logger.setLevel(logging.INFO)
    
    # Create a handler to print log messages to the console (standard output).
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter to define the structure of our log messages.
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the configured handler to the logger.
    logger.addHandler(handler)
    
    return logger