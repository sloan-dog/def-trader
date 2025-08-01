"""
Wrapper logger for consistent logging across the application.

This module provides a simplified interface to loguru with automatic
exception handling and environment-aware formatting.
"""
import sys
import os
from typing import Any, Optional
from functools import wraps
from loguru import logger as _loguru_logger

from src.shared_logging import setup_logging, log_exception as _log_exception


class AppLogger:
    """
    Application logger wrapper that provides a clean interface to loguru.
    
    Features:
    - Automatic exception capture in error() when in exception context
    - Environment-aware formatting (JSON in deployment, human-readable locally)
    - Singleton pattern to ensure single configuration
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logger with environment-appropriate settings."""
        if not self._initialized:
            # Determine if we're in a deployment environment
            is_deployment = bool(
                os.environ.get("K_SERVICE") or 
                os.environ.get("GOOGLE_CLOUD_PROJECT") or
                os.environ.get("DEPLOYMENT_ENV", "").lower() in ["prod", "production", "staging"]
            )
            
            # Setup logging with automatic JSON detection
            setup_logging(
                level=os.environ.get("LOG_LEVEL", "INFO"),
                enable_json=is_deployment,
                app_name=os.environ.get("K_SERVICE", os.environ.get("APP_NAME", "signaler"))
            )
            
            self._initialized = True
    
    def _is_in_exception_context(self) -> bool:
        """Check if we're currently in an exception context."""
        return sys.exc_info()[0] is not None
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message."""
        _loguru_logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        _loguru_logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message."""
        _loguru_logger.warning(message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """
        Log an error message.
        
        If called within an exception context (inside except block),
        automatically captures the full stack trace.
        
        Args:
            message: Error message
            exception: Optional exception object (deprecated, auto-detected)
            **kwargs: Additional context
        """
        if self._is_in_exception_context() or exception is not None:
            # Use the shared logging exception handler
            _log_exception(message, exception, **kwargs)
        else:
            _loguru_logger.error(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """
        Log an exception with full traceback.
        
        This is equivalent to error() when in exception context.
        """
        _log_exception(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical message."""
        if self._is_in_exception_context():
            _loguru_logger.opt(exception=True).critical(message, **kwargs)
        else:
            _loguru_logger.critical(message, **kwargs)
    
    def bind(self, **kwargs) -> Any:
        """Bind contextual data to the logger."""
        return _loguru_logger.bind(**kwargs)
    
    def opt(self, **kwargs) -> Any:
        """
        Configure logger options.
        
        Note: exception=True is automatically handled in error() method.
        """
        return _loguru_logger.opt(**kwargs)
    
    def catch(self, *args, **kwargs) -> Any:
        """Decorator/context manager to catch and log exceptions."""
        return _loguru_logger.catch(*args, **kwargs)


# Create singleton instance
logger = AppLogger()

# Convenience function for module-level setup
def setup_module_logging(
    level: str = "INFO",
    enable_json: Optional[bool] = None,
    app_name: Optional[str] = None
) -> None:
    """
    Setup logging for a module or application.
    
    This is useful for entry points that need custom configuration.
    
    Args:
        level: Log level
        enable_json: Force JSON output (None = auto-detect)
        app_name: Application name for context
    """
    setup_logging(level=level, enable_json=enable_json, app_name=app_name)
    # Reinitialize our logger
    AppLogger._initialized = False
    logger.__init__()