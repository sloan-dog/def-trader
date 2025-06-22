"""
BigQuery utilities and decorators.
"""
import time
from functools import wraps
from typing import Callable, TypeVar, Any
from google.cloud.exceptions import GoogleCloudError
from loguru import logger

T = TypeVar('T')


def retry_on_error(max_retries: int = 3, delay: int = 5) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry BigQuery operations on failure.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except GoogleCloudError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay * (attempt + 1))
            # This should never be reached due to the raise in the except block
            raise RuntimeError("Retry logic error")
        return wrapper
    return decorator


def validate_table_name(table_name: str) -> bool:
    """
    Validate BigQuery table name.

    Args:
        table_name: Table name to validate

    Returns:
        True if valid, False otherwise
    """
    import re
    # BigQuery table names must:
    # - Start with a letter or underscore
    # - Contain only letters, numbers, and underscores
    # - Be at most 1024 characters
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]{0,1023}$'
    return bool(re.match(pattern, table_name))


def format_table_id(project_id: str, dataset_id: str, table_name: str) -> str:
    """
    Format full table ID for BigQuery.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_name: Table name

    Returns:
        Formatted table ID
    """
    return f"{project_id}.{dataset_id}.{table_name}"