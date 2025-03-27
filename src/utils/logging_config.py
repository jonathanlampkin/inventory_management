import logging
import logging.config
import os
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    app_name: str = "inventory_management"
) -> logging.Logger:
    """
    Set up structured logging with file and console handlers.
    
    Parameters:
    -----------
    log_dir : str
        Directory to store log files
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    app_name : str
        Application name for logger
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")
    
    # Define logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s | %(levelname)s | %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': log_file,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': os.path.join(log_dir, f"{app_name}_errors_{timestamp}.log"),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file', 'error_file'],
                'level': log_level,
                'propagate': True
            },
            'inventory_management': {
                'handlers': ['console', 'file', 'error_file'],
                'level': log_level,
                'propagate': False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get logger
    logger = logging.getLogger(app_name)
    
    # Log startup message
    logger.info(f"Logging initialized: {app_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger

class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[dict] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Add context information to log messages."""
        # Add timestamp if not present
        if 'timestamp' not in self.extra:
            self.extra['timestamp'] = datetime.now().isoformat()
        
        # Format message with context
        context_str = ' | '.join(f"{k}={v}" for k, v in self.extra.items())
        return f"{msg} | {context_str}", kwargs

def get_logger(
    name: str,
    context: Optional[dict] = None
) -> LoggerAdapter:
    """
    Get a logger with optional context.
    
    Parameters:
    -----------
    name : str
        Logger name
    context : dict, optional
        Additional context to include in log messages
    
    Returns:
    --------
    LoggerAdapter
        Logger adapter with context
    """
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)

# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Function {func.__name__} completed in {duration:.2f} seconds",
                    extra={'duration': duration}
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"Function {func.__name__} failed after {duration:.2f} seconds: {str(e)}",
                    extra={'duration': duration, 'error': str(e)},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator 