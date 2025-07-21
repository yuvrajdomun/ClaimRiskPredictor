"""
Logging Utility Module

This module provides centralized logging configuration for the insurance
fraud detection system. It sets up structured logging with file rotation
and different log levels.

Functions:
    setup_logger: Configure and return logger instance
    get_logger: Get existing logger instance
    
Example:
    >>> from insurance_fraud_detector.utils.logger import setup_logger
    >>> logger = setup_logger('my_module')
    >>> logger.info('This is an info message')
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    
    This formatter converts log records to JSON format for better
    parsing and analysis in log management systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log message
        """
        log_obj = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_obj.update(record.extra_fields)
        
        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """
    Colored console formatter for better readability during development.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Colored log message
        """
        # Apply color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            record.levelname = colored_levelname
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


def setup_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    json_format: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    This function creates a logger with both file and console handlers,
    configurable formatting, and automatic log rotation.
    
    Args:
        name: Name of the logger (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, creates default log file
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Whether to use JSON formatting
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger('fraud_detector', level='DEBUG')
        >>> logger.info('Logger configured successfully')
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set logger level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger instance.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_logging_from_config(config: dict) -> None:
    """
    Configure logging from configuration dictionary.
    
    Args:
        config: Logging configuration dictionary
    """
    logging_config = config.get('logging', {})
    
    # Set up root logger
    root_logger = setup_logger(
        name='insurance_fraud_detector',
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('file_path'),
        max_file_size=logging_config.get('max_file_size', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5),
        json_format=logging_config.get('json_format', False)
    )
    
    # Set logging level for third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    root_logger.info('Logging configured successfully')


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    This mixin automatically creates a logger instance based on the
    class name and provides easy access to logging methods.
    
    Example:
        >>> class MyClass(LoggerMixin):
        ...     def do_something(self):
        ...         self.logger.info('Doing something')
    """
    
    @property
    def logger(self) -> logging.Logger:
        """
        Get logger instance for this class.
        
        Returns:
            Logger instance named after the class
        """
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__module__}.{self.__class__.__name__}")
        return self._logger


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
        
    Example:
        >>> @log_execution_time
        ... def my_function():
        ...     # Function implementation
        ...     pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Function {func.__name__} executed successfully in {execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.3f}s: {str(e)}"
            )
            raise
    
    return wrapper


def log_method_calls(cls):
    """
    Class decorator to log all method calls.
    
    Args:
        cls: Class to decorate
        
    Returns:
        Decorated class
        
    Example:
        >>> @log_method_calls
        ... class MyClass:
        ...     def my_method(self):
        ...         pass
    """
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr))
    return cls