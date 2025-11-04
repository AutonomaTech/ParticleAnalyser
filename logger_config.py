
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

# --- Add a dedicated, filtered error logger ---
def specific_error_filter(record):
    """
    Only log to error.log if the 'should_log_to_error_file' key in extra dict is True.
    """
    # record["extra"] includes all data that will be flagged as  extra
    return record["extra"].get("process_error", False)


# --- Add a dedicated error logger ---
# This log file will append all ERROR level messages and won't rotate.
logger.add(
    os.path.join(LOG_DIR, "error.log"),
    level="ERROR",  # This filter ensures only ERROR and above levels are caught
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {process} >>> {message}",
    encoding="utf-8",
    rotation=None,        # No rotation
    retention=None,       # No deletion
    compression=None      # No compression
)
def get_logger(name: str):
    return logger.bind(name=name)

# This ensures the logger is imported and configured in the main application
__all__ = ["logger", "get_logger"]
def get_logger(name: str):
    return logger.bind(name=name)

# This ensures the logger is imported and configured in the main application
__all__ = ["logger", "get_logger"]
