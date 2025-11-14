import logging
import sys
from pathlib import Path

def setup_logging(log_file_name: str = "pipeline.log"):
    """
    Sets up a centralized logger for the project.

    This configuration directs log messages to both a file and the standard
    output (console). It establishes a consistent format for all log entries.

    Args:
        log_file_name (str): The name of the log file to be created.
                             Defaults to "pipeline.log".
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / log_file_name

    # Define the logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set the minimum level for the root logger

    # Clear any existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO) # Log INFO and above to the file
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create a stream handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO) # Log INFO and above to the console
    stream_handler.setFormatter(logging.Formatter(log_format))
    
    # Add handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info("📝 Logging configured successfully.") 