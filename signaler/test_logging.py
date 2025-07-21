#!/usr/bin/env python3
"""
Test script to verify structured JSON logging works correctly.
"""
import sys
import os
import traceback
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.shared_logging import setup_logging, log_exception, log_with_context
from loguru import logger


def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic logging...")
    
    logger.info("This is a basic info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")


def test_structured_logging():
    """Test structured logging with extra fields."""
    print("Testing structured logging...")
    
    logger.info("Processing ticker data", 
                ticker="AAPL",
                records_processed=150,
                success_rate=0.95)
    
    logger.warning("API rate limit approaching",
                   api_calls_made=45,
                   rate_limit=50,
                   time_remaining=300)
    
    logger.error("Failed to fetch data",
                 ticker="TSLA",
                 error_type="API_ERROR",
                 retry_count=3)


def test_exception_logging():
    """Test exception logging."""
    print("Testing exception logging...")
    
    try:
        # Simulate an error
        raise ValueError("This is a test exception")
    except Exception as e:
        log_exception("Failed to process data", exception=e, ticker="AAPL")


def test_nested_exception():
    """Test nested exception handling."""
    print("Testing nested exception...")
    
    try:
        try:
            raise ValueError("Inner exception")
        except Exception as inner_e:
            raise RuntimeError("Outer exception") from inner_e
    except Exception as e:
        log_exception("Complex error occurred", exception=e, context="data_processing")


def main():
    """Run all logging tests."""
    print("Setting up logging...")
    
    # Configure logging with JSON format
    setup_logging(
        level="DEBUG",
        enable_json=True
    )
    
    print("Running logging tests...")
    print("=" * 50)
    
    test_basic_logging()
    print()
    
    test_structured_logging()
    print()
    
    test_exception_logging()
    print()
    
    test_nested_exception()
    print()
    
    print("=" * 50)
    print("Logging tests completed. Check the output above for JSON logs.")


if __name__ == "__main__":
    main() 