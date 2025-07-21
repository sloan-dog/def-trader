"""
Shared logging module for all Cloud Run apps in the monorepo.

This provides consistent, structured JSON logging that works well with Google Cloud Logging.
"""
import sys
import json
import traceback
from loguru import logger
from datetime import datetime
import os
from typing import Any, Dict, Optional


def create_structured_log(record: Dict[str, Any]) -> str:
    """
    Create a structured log entry compatible with Google Cloud Logging.
    
    This creates a single JSON object that Google Cloud will treat as one log entry.
    
    Args:
        record: Loguru record dictionary
        
    Returns:
        JSON string representation of the log entry
    """
    # Base log structure
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "severity": record["level"].name,
        "message": record["message"],
        "sourceLocation": {
            "file": record["file"].name,
            "line": record["line"],
            "function": record["function"]
        }
    }
    
    # Add exception information if present
    if record["exception"] is not None:
        exc_type, exc_value, exc_traceback = record["exception"]
        
        # Create structured exception info
        exception_info = {
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        }
        
        log_entry["exception"] = exception_info
        
        # Also append the full exception to the message for visibility
        log_entry["message"] = f"{record['message']}\n\nException: {exc_type.__name__}: {exc_value}\nTraceback:\n{exception_info['traceback']}"
    
    # Add any extra fields
    if record["extra"]:
        log_entry["extra"] = record["extra"]
    
    # Add module name
    log_entry["module"] = record["name"]
    
    return json.dumps(log_entry, default=str, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    enable_json: bool = None,
    log_file: str = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    app_name: str = None
):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_json: Enable JSON logging format. If None, auto-detect based on environment
        log_file: Optional log file path for file logging
        rotation: Log file rotation policy
        retention: Log file retention policy
        app_name: Name of the application for logging context
    """
    # Remove default logger
    logger.remove()
    
    # Auto-detect if we're running in Google Cloud Run
    if enable_json is None:
        # Check for Google Cloud Run environment variables
        enable_json = bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
    
    # Configure stderr output
    if enable_json:
        # JSON format for Google Cloud Logging
        def json_sink(message):
            try:
                # Add app name to the log record if provided
                if app_name:
                    message.record["extra"]["app_name"] = app_name
                
                log_json = create_structured_log(message.record)
                # Write to stderr with newline to ensure it's treated as one log entry
                sys.stderr.write(log_json + "\n")
                sys.stderr.flush()  # Ensure immediate output
            except Exception as e:
                # Fallback to simple format if JSON creation fails
                fallback_msg = f"{message.record['time'].isoformat()} | {message.record['level'].name} | {message.record['message']}"
                sys.stderr.write(fallback_msg + "\n")
                sys.stderr.flush()
        
        logger.add(
            json_sink,
            level=level,
            backtrace=True,
            diagnose=True,
            enqueue=True,
            catch=True
        )
    else:
        # Human-readable format for local development
        format_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level>"
        if app_name:
            format_str += f" | <cyan>{app_name}</cyan>"
        format_str += " | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        
        logger.add(
            sys.stderr,
            format=format_str,
            level=level,
            backtrace=True,
            diagnose=True,
            enqueue=True,
            catch=True
        )
    
    # Add file logging if specified
    if log_file:
        if enable_json:
            def json_file_sink(message):
                try:
                    # Add app name to the log record if provided
                    if app_name:
                        message.record["extra"]["app_name"] = app_name
                    
                    log_json = create_structured_log(message.record)
                    with open(log_file, 'a') as f:
                        f.write(log_json + "\n")
                except Exception:
                    # Fallback to simple format
                    fallback_msg = f"{message.record['time'].isoformat()} | {message.record['level'].name} | {message.record['message']}"
                    with open(log_file, 'a') as f:
                        f.write(fallback_msg + "\n")
            
            logger.add(
                json_file_sink,
                level=level,
                rotation=rotation,
                retention=retention,
                backtrace=True,
                diagnose=True,
                enqueue=True
            )
        else:
            format_str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8}"
            if app_name:
                format_str += f" | {app_name}"
            format_str += " | {name}:{function}:{line} - {message}"
            
            logger.add(
                log_file,
                format=format_str,
                level=level,
                rotation=rotation,
                retention=retention,
                backtrace=True,
                diagnose=True,
                enqueue=True
            )
    
    logger.info(f"Logging configured: level={level}, json_format={enable_json}, file={log_file}, app_name={app_name}")


def log_exception(message: str, exception: Exception = None, **kwargs):
    """
    Log an exception with full traceback information.
    
    Args:
        message: Error message
        exception: Exception object (if None, will get from sys.exc_info())
        **kwargs: Additional context to include in the log
    """
    # Use opt(exception=True) to capture the current exception context with full traceback
    logger.opt(exception=True).error(message, **kwargs)


def log_with_context(message: str, level: str = "INFO", **kwargs):
    """
    Log a message with additional context in structured format.
    
    Args:
        message: Log message
        level: Log level
        **kwargs: Additional context to include
    """
    if level.upper() == "DEBUG":
        logger.debug(message, **kwargs)
    elif level.upper() == "INFO":
        logger.info(message, **kwargs)
    elif level.upper() == "WARNING":
        logger.warning(message, **kwargs)
    elif level.upper() == "ERROR":
        logger.error(message, **kwargs)
    elif level.upper() == "CRITICAL":
        logger.critical(message, **kwargs)
    else:
        logger.info(message, **kwargs)


def is_cloud_run() -> bool:
    """
    Check if running in Google Cloud Run environment.
    
    Returns:
        True if running in Cloud Run, False otherwise
    """
    return bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_PROJECT"))


def get_app_name() -> str:
    """
    Get the application name from environment variables.
    
    Returns:
        Application name or 'unknown' if not set
    """
    return os.environ.get("K_SERVICE", os.environ.get("APP_NAME", "unknown")) 