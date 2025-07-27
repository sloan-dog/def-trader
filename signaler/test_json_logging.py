#!/usr/bin/env python3
"""Test script to verify JSON logging for exceptions."""

import os
import sys

# Simulate cloud environment
os.environ["K_SERVICE"] = "test-service"

# Import and setup logging
from src.shared_logging import setup_logging

# Setup logging with JSON format
setup_logging(app_name="test-json-logging")

# Test 1: Regular log messages
from loguru import logger
logger.info("This is a regular info message")
logger.error("This is an error message")

# Test 2: Logged exception
try:
    1 / 0
except Exception as e:
    logger.exception("Caught exception")

# Test 3: Uncaught exception (this will trigger the custom exception hook)
print("\nNow testing uncaught exception...")
import torch  # This will cause ModuleNotFoundError