import logging
import os
import sys
from datetime import datetime

def loggers(log_file_path='log.txt', level=logging.INFO):
    """
    Configures and returns a logger instance that logs to both console and file.
    Ensures only one set of handlers is added to the root logger.

    Args:
        log_file_path (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Use the root logger to avoid duplicate messages if called multiple times
    logger = logging.getLogger()

    # Check if handlers have already been added to this logger
    if not logger.handlers:
        logger.setLevel(level)  # Set the overall minimum logging level

        # --- File Handler ---
        # Ensure directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create file handler
        # Use 'a' mode to append if file exists, 'w' to overwrite
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(level)  # Log level for the file

        # --- Console Handler ---
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)  # Log level for the console

        # --- Formatter ---
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # --- Add Handlers ---
        logger.addHandler(fh)
        logger.addHandler(ch)

        # Optional: Prevent propagation to ancestor loggers if needed
        # logger.propagate = False

        # logger.info(f"Logger configured. Logging to console and file: {log_file_path}") # Log initial setup

    # else:
    #     logger.debug("Logger already configured.") # Debug message if handlers exist

    return logger

# Example usage:
if __name__ == '__main__':
    # Configure logger (call this early in your main script)
    log_path = os.path.join('logs', 'test_log.txt') # Example path
    my_logger = loggers(log_path, level=logging.DEBUG)

    # Use the logger
    my_logger.debug("This is a debug message.")
    my_logger.info("This is an info message.")
    my_logger.warning("This is a warning message.")
    my_logger.error("This is an error message.")
    my_logger.critical("This is a critical message.")

    # Logger from another module will use the same configuration
    another_logger = logging.getLogger("module_b")
    another_logger.info("Message from another module.")