"""
This module provides a flexible logging system with customizable log levels and handlers.

Features:
- Multiple log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE
- File logging with configurable file path and rotation
- Resource management for proper cleanup of file handlers

Note:
When using file handlers, it's important to properly close them when they're no longer needed
to avoid file lock issues, especially on Windows. Use the close_logger() function to ensure
all handlers are properly closed and resources are released.
"""
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Type
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Default logger instance
DEFAULT_LOGGER = None

class LoggerConfig:
    """Configuration class for logger settings"""
    DEFAULT_NAME = "PyOmnix"
    DEFAULT_LEVEL = logging.INFO
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Custom log levels
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

class OmnixLogger(logging.Logger):
    """Extends the standard Logger with trace-level logging capabilities."""
    def trace(self, msg, *args, **kwargs):
        """Log a message with TRACE level."""
        if self.isEnabledFor(LoggerConfig.TRACE):
            self.log(LoggerConfig.TRACE, msg, *args, **kwargs)

    def validate(self, condition: bool, message: str, exception_type: Type[Exception] = AssertionError, log_level: int = logging.ERROR) -> None:
        """
        Validate a condition and log an error if it fails.

        Args:
            condition: Condition to validate
            message: Error message if condition fails
            exception_type: Type of exception to raise
            logger_name: Name of logger to use (None for default)
            log_level: Log level to use

        Raises:
            exception_type: If condition is False
        """
        if not condition:
            self.log(log_level, message)
            raise exception_type(message)

# use the custom logger class to replace the default logger class
logging.setLoggerClass(OmnixLogger)

def setup_logger(*,
    name: str = LoggerConfig.DEFAULT_NAME,
    log_level: int = LoggerConfig.DEFAULT_LEVEL,
    log_file: Optional[Path | str] = None,
    log_format: str = LoggerConfig.DEFAULT_FORMAT,
    date_format: str = LoggerConfig.DEFAULT_DATE_FORMAT,
    propagate: bool = False,
    add_trace_level: bool = True,
    rotation: str = None, # Rotation type: 'size' or 'time'
    max_size: int = 10 * 1024 * 1024, # 10 MB
    backup_count: int = 3,
    interval: int = 1 # Days
) -> logging.Logger:
    """
    Configure and return a logger instance with consistent formatting.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional path to log file
        log_format: Format string for log messages
        date_format: Format string for date in log messages
        propagate: Whether to propagate messages to parent loggers, default is False to make loggers independent
        add_trace_level: Whether to add trace method to logger
    
    Returns:
        Configured logger instance
    """
    new_logger = logging.getLogger(name)
    new_logger.setLevel(log_level)
    new_logger.propagate = propagate
    
    # Clear any existing handlers
    for handler in list(new_logger.handlers):
        handler.close()  # Ensure file handlers are properly closed
        new_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    new_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file is not None:
        try:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Choose the appropriate file handler based on rotation type
            if rotation == 'size':
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=max_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            elif rotation == 'time':
                file_handler = TimedRotatingFileHandler(
                    log_file,
                    when='D',  # Daily rotation
                    interval=interval,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            else:
                # Default to standard FileHandler if no rotation specified
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                
            file_handler.setFormatter(formatter)
            new_logger.addHandler(file_handler)
        except (IOError, PermissionError) as e:
            original_level = console_handler.level
            try:
                console_handler.setLevel(logging.WARNING)
                new_logger.warning("Failed to create log file at %s: %s", str(log_file), str(e))
            finally:
                console_handler.setLevel(original_level)
    
    # Add trace method if requested
    if add_trace_level and not hasattr(new_logger, 'trace'):
        def trace_method(msg, *args, **kwargs):
            """Log a message with TRACE level."""
            if new_logger.isEnabledFor(LoggerConfig.TRACE):
                new_logger.log(LoggerConfig.TRACE, msg, *args, **kwargs)
        
        new_logger.trace = trace_method
    
    # Store as default logger if it's the first one created
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = new_logger
    
    return new_logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get an existing logger or create a new one if it doesn't exist.
    
    Args:
        name: Logger name (if None, returns the default logger)
        
    Returns:
        Logger instance
    """
    if name is None:
        global DEFAULT_LOGGER
        if DEFAULT_LOGGER is None:
            DEFAULT_LOGGER = setup_logger()
        return DEFAULT_LOGGER
    
    new_logger = logging.getLogger(name)
    if not new_logger.handlers:
        # If logger exists but has no handlers, set it up
        return setup_logger(name=name)
    return new_logger

class ExceptionHandler:
    """Handles exceptions and logs them appropriately"""
    
    # Map exception types to their log levels
    EXCEPTION_LEVEL_MAP: Dict[Type[Exception], int] = {
        AssertionError: logging.ERROR,
        ValueError: logging.ERROR,
        TypeError: logging.ERROR,
        KeyError: logging.ERROR,
        FileNotFoundError: logging.ERROR,
        PermissionError: logging.ERROR,
        # Add more exception types as needed
    }
    
    # Default level for unmapped exceptions
    DEFAULT_LEVEL = logging.ERROR
    EXIT_ON_CRITICAL = True
    
    @staticmethod
    def get_log_level(exc_type: Type[Exception]) -> int:
        """Get the appropriate log level for an exception type"""
        if not issubclass(exc_type, Exception):
            return logging.CRITICAL
        for exception_class, level in ExceptionHandler.EXCEPTION_LEVEL_MAP.items():
            if issubclass(exc_type, exception_class):
                return level
        return ExceptionHandler.DEFAULT_LEVEL
    
    @staticmethod
    def format_exception(exc_type: Type[Exception], exc_value: Exception, exc_traceback) -> str:
        """Format exception information into a string"""
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return ''.join(tb_lines)

def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook that logs exceptions based on their type.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    new_logger = get_logger()
    log_level = ExceptionHandler.get_log_level(exc_type)
    
    # Format the exception message
    exc_msg = f"{exc_type.__name__}: {exc_value}"
    
    # Log with appropriate level and include traceback
    new_logger.log(log_level, exc_msg, exc_info=(exc_type, exc_value, exc_traceback))
    
    # For critical errors, we might want to exit the program
    if log_level >= logging.CRITICAL and ExceptionHandler.EXIT_ON_CRITICAL:
        sys.exit(1)

# Override the default excepthook
sys.excepthook = custom_excepthook


def close_logger(name: Optional[str] = None) -> None:
    """
    Close all handlers for a logger to release file resources.
    
    Args:
        name: Logger name (if None, closes handlers for the default logger)
    """
    if name is None:
        global DEFAULT_LOGGER
        if DEFAULT_LOGGER is not None:
            for handler in list(DEFAULT_LOGGER.handlers):
                handler.close()
                DEFAULT_LOGGER.removeHandler(handler)
    else:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

# Initialize the default logger
default_logger = get_logger()

# Example usage
if __name__ == "__main__":
    logger = setup_logger(name="test", log_level=logging.DEBUG, log_file="test.log")
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
