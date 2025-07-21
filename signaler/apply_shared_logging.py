#!/usr/bin/env python3
"""
Script to apply shared logging to Cloud Run apps.

This script helps migrate existing apps to use the shared logging module.
"""
import os
import re
import sys
from pathlib import Path


def find_python_files(directory):
    """Find all Python files in a directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and .git directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.venv']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def update_imports(file_path, shared_logging_path):
    """Update imports in a Python file to use shared logging."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Calculate relative path to shared_logging.py
    file_dir = os.path.dirname(file_path)
    rel_path = os.path.relpath(shared_logging_path, file_dir)
    
    # Replace old logging imports
    old_imports = [
        r'from src\.utils\.logging_config import setup_logging, log_exception',
        r'from src\.utils\.logging_config import setup_logging',
        r'from src\.utils\.logging_config import log_exception',
        r'from shared\.logging import setup_logging, log_exception',
        r'from shared\.logging import setup_logging',
        r'from shared\.logging import log_exception',
    ]
    
    new_import = f"""from src.shared_logging import setup_logging, log_exception"""
    
    modified = False
    for old_import in old_imports:
        if re.search(old_import, content):
            content = re.sub(old_import, new_import, content)
            modified = True
    
    if modified:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated: {file_path}")
        return True
    
    return False


def main():
    """Main function to apply shared logging to apps."""
    if len(sys.argv) < 2:
        print("Usage: python apply_shared_logging.py <app_directory>")
        print("Example: python apply_shared_logging.py ../another-app")
        sys.exit(1)
    
    app_dir = sys.argv[1]
    if not os.path.exists(app_dir):
        print(f"Error: Directory {app_dir} does not exist")
        sys.exit(1)
    
    # Path to shared_logging.py (relative to this script)
    shared_logging_path = os.path.join(os.path.dirname(__file__), 'src', 'shared_logging.py')
    
    if not os.path.exists(shared_logging_path):
        print(f"Error: shared_logging.py not found at {shared_logging_path}")
        sys.exit(1)
    
    print(f"Applying shared logging to: {app_dir}")
    print(f"Shared logging module: {shared_logging_path}")
    print()
    
    # Find all Python files
    python_files = find_python_files(app_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Update imports
    updated_count = 0
    for file_path in python_files:
        if update_imports(file_path, shared_logging_path):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} files")
    
    # Create a simple test script for the app
    test_script = os.path.join(app_dir, 'test_shared_logging.py')
    with open(test_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Test script for shared logging in this app.
"""
import sys
import os

from src.shared_logging import setup_logging, log_exception
from loguru import logger

def test_logging():
    """Test the shared logging functionality."""
    print("Testing shared logging...")
    
    # Configure logging
    setup_logging(level="DEBUG", app_name="test-app")
    
    # Test basic logging
    logger.info("This is a test info message")
    logger.warning("This is a test warning")
    logger.error("This is a test error")
    
    # Test structured logging
    logger.info("Processing data", 
               records_processed=100,
               success_rate=0.95)
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        log_exception("Test exception occurred", exception=e)
    
    print("Logging test completed!")

if __name__ == "__main__":
    test_logging()
''')
    
    print(f"Created test script: {test_script}")
    print("\nTo test the logging, run:")
    print(f"cd {app_dir}")
    print("python test_shared_logging.py")


if __name__ == "__main__":
    main() 