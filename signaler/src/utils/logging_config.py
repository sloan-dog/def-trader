"""
Centralized logging configuration for Google Cloud Run compatibility.
"""
import sys
import json
import traceback
from loguru import logger
from datetime import datetime
import os


def serialize(record):
    """Serialize log record to JSON format for Google Cloud Logging."""
    # Extract exception info if present
    exception_info = None
    if record["exception"] is not None:
        # Get the full traceback
        tb_lines = traceback.format_exception(
            record["exception"].type,
            record["exception"].value,
            record["exception"].traceback
        )
        exception_info = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": "".join(tb_lines)
        }
    
    # Map to Google Cloud Logging severity levels
    severity_mapping = {
        "TRACE": "DEBUG",
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "SUCCESS": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL"
    }
    
    # Create Google Cloud compatible log entry
    gcp_entry = {
        "message": record["message"],
        "severity": severity_mapping.get(record["level"].name, "INFO"),
        "timestamp": record["time"].isoformat(),
        "labels": {
            "file": record["file"].name,
            "function": record["function"],
            "line": str(record["line"])
        }
    }
    
    # Add structured data
    if exception_info:
        gcp_entry["exception"] = exception_info
        # Also add the full traceback to the message for visibility
        gcp_entry["message"] = f"{record['message']}\\n\\nException: {exception_info['type']}: {exception_info['value']}\\nTraceback:\\n{exception_info['traceback']}"
    
    if record["extra"]:
        gcp_entry["jsonPayload"] = record["extra"]
    
    # Return the serialized format wrapped in the proper message structure
    return json.dumps(gcp_entry, default=str, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    enable_json: bool = None,
    log_file: str = None,
    rotation: str = "1 day",
    retention: str = "30 days"
):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_json: Enable JSON logging format. If None, auto-detect based on environment
        log_file: Optional log file path for file logging
        rotation: Log file rotation policy
        retention: Log file retention policy
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
            record = message.record
            sys.stderr.write(serialize(record) + "\n")
        
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
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
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
                with open(log_file, 'a') as f:
                    f.write(serialize(message.record) + "\n")
            
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
            logger.add(
                log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level=level,
                rotation=rotation,
                retention=retention,
                backtrace=True,
                diagnose=True,
                enqueue=True
            )
    
    logger.info(f"Logging configured: level={level}, json_format={enable_json}, file={log_file}")


# Convenience function to log exceptions with full traceback
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