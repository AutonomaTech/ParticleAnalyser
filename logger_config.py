from loguru import logger
import os
import sys

# Configure logger
LOG_FORMAT = "{name}: {time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {process} >>> {message}"
logger.remove()  # Remove default logger
logger.add(sys.stdout, level="DEBUG")  # Add a new logger to stdout with DEBUG level
script_dir = os.path.dirname(sys.argv[0])
os.chdir(script_dir)

# Now use the script directory for the log file
log_file_path = os.path.join(script_dir, "file.log")

logger.add(log_file_path,
           rotation="1 week",  # Rotate logs every week
           retention="10 days",  # Keep logs for 10 days
           compression="zip",  # Compress logs with zip
           level="DEBUG",  # Set the minimum level to log
           format="{time} {level} {message}")  # Custom format

def get_logger(name: str):
    return logger.bind(name=name)

# This ensures the logger is imported and configured in the main application
__all__ = ["logger", "get_logger"]