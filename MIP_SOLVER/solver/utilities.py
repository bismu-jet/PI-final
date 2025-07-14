import logging

def setup_logger():
    """
    Sets up a logger that avoids adding duplicate handlers.
    """
    logger = logging.getLogger("MIPSolver")

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    
    # Create a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(ch)
    
    return logger