import logging

def setup_logger():
    """
    Sets up a logger that avoids adding duplicate handlers.
    """
    logger = logging.getLogger("MIPSolver")
    
    # --- THIS IS THE FIX ---
    # Check if the logger already has handlers configured.
    # If it does, it means we've already set it up, so we just return it.
    if logger.hasHandlers():
        return logger
    # ----------------------

    logger.setLevel(logging.INFO)
    
    # Create a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(ch)
    
    return logger