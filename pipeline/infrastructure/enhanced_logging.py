"""
Enhanced Logging Configuration for Production Robustness
Provides structured logging with performance metrics and error tracking
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Dict
from contextlib import contextmanager


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record):
        record.timestamp = datetime.utcnow().isoformat()
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": getattr(record, 'timestamp', datetime.utcnow().isoformat()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'performance_metrics'):
            log_entry['performance_metrics'] = record.performance_metrics
            
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_enhanced_logging(
    level: str = "INFO",
    enable_structured: bool = False,
    log_file: str = None
) -> logging.Logger:
    """
    Setup enhanced logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_structured: Whether to use structured JSON logging
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger("agentic_startup_studio")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    if enable_structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(PerformanceFilter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter() if enable_structured else console_formatter)
        file_handler.addFilter(PerformanceFilter())
        logger.addHandler(file_handler)
    
    return logger


@contextmanager
def log_performance(logger: logging.Logger, operation: str, **context):
    """
    Context manager to log operation performance metrics.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation being measured
        **context: Additional context to include in logs
    """
    start_time = time.time()
    start_timestamp = datetime.utcnow().isoformat()
    
    try:
        logger.info(f"Starting {operation}", extra={
            'performance_metrics': {'start_time': start_timestamp},
            **context
        })
        
        yield
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Completed {operation}", extra={
            'performance_metrics': {
                'start_time': start_timestamp,
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': duration,
                'status': 'success'
            },
            **context
        })
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        logger.error(f"Failed {operation}: {str(e)}", extra={
            'performance_metrics': {
                'start_time': start_timestamp,
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': duration,
                'status': 'error',
                'error': str(e)
            },
            **context
        }, exc_info=True)
        
        raise


# Global logger instance
_enhanced_logger = None


def get_enhanced_logger() -> logging.Logger:
    """Get the global enhanced logger instance."""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = setup_enhanced_logging()
    return _enhanced_logger