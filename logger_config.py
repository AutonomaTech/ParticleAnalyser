
from loguru import logger
import sys
import os

# Configure logger
LOG_FORMAT = "{name}: {time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {process} >>> {message}"

# Ensure log directory exists
LOG_DIR = r"D:\AVANATICA Q\Logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default logger
logger.remove()

# Add console logger (stdout) - logs all levels
logger.add(sys.stdout, level="DEBUG", format=LOG_FORMAT)

# Add file logger with daily rotation and permanent retention - logs all levels
logger.add(
    os.path.join(LOG_DIR, "app_{time:YYYY-MM-DD}.log"),  # One file per day with date in filename
    rotation="00:00",      # Rotate at midnight every day
    retention=None,        # Keep all log files permanently (no deletion)
    compression="zip",     # Compress old log files
    level="DEBUG",         # Log all levels (DEBUG and above)
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {process} >>> {message}",  # File log format
    encoding="utf-8"       # Ensure proper display of non-ASCII characters
)

def get_logger(name: str):
    return logger.bind(name=name)

# This ensures the logger is imported and configured in the main application
__all__ = ["logger", "get_logger"]