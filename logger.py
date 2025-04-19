import logging
import os
import sys
from datetime import datetime

def loggers(log_file_path='log.txt', level=logging.INFO):
    """
    Configures and returns a logger instance that logs to both console and file.
    Creates a distinct logger based on the file path to avoid handler duplication.

    Args:
        log_file_path (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Use a unique name for the logger based on the file path to prevent conflicts
    logger_name = log_file_path.replace('/', '_').replace('\\', '_').replace('.', '_')
    logger = logging.getLogger(logger_name)

    # Check if handlers have already been added to THIS specific logger
    if not logger.handlers:
        logger.setLevel(level)  # Set the overall minimum logging level for this logger

        # --- File Handler ---
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        fh.setLevel(level)

        # --- Console Handler ---
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)

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

        # Prevent propagation to the root logger to avoid duplicate console output
        # if multiple loggers are configured via this function.
        logger.propagate = False

        logger.debug(f"Logger '{logger_name}' configured. Logging to: {log_file_path}")

    # else:
    #      logger.debug(f"Logger '{logger_name}' already configured.")

    return logger

# Example usage:
if __name__ == '__main__':
    # Configure logger (call this early in your main script)
    log_path1 = os.path.join('logs', 'test_log1.txt')
    my_logger1 = loggers(log_path1, level=logging.DEBUG)

    # Use the logger
    my_logger1.debug("This is logger 1 debug.")
    my_logger1.info("This is logger 1 info.")

    # Configure another logger for a different file
    log_path2 = os.path.join('logs', 'test_log2.txt')
    my_logger2 = loggers(log_path2, level=logging.INFO)
    my_logger2.info("This is logger 2 info.")
    my_logger2.warning("This is logger 2 warning.")

    # Getting logger by name works as expected
    logger_ref1 = logging.getLogger(log_path1.replace('/', '_').replace('\\', '_').replace('.', '_'))
    logger_ref1.info("Message via logger reference 1.") # Should go to file1 and console

    # Root logger (if configured elsewhere) might still log if propagation isn't stopped
    # logging.info("This might be logged by root logger if configured.")