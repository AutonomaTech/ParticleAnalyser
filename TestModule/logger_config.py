from loguru import logger
import sys

# Configure logger
logger.remove()  # Remove default logger
logger.add(sys.stdout, level="DEBUG")  # Add a new logger to stdout with DEBUG level
logger.add("file.log",
           rotation="1 week",  # Rotate logs every week
           retention="10 days",  # Keep logs for 10 days
           compression="zip",  # Compress logs with zip
           level="DEBUG",  # Set the minimum level to log
           format="{time} {level} {message}")  # Custom format

def get_logger(name: str):
    return logger.bind(name=name)

# This ensures the logger is imported and configured in the main application
__all__ = ["logger", "get_logger"]