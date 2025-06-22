"""
BigQuery utilities module.

This module provides a modular interface to Google BigQuery operations,
split into focused components for better maintainability.
"""

# Main client interface (maintains backward compatibility)
from .client import BigQueryClient

# Individual components for advanced usage
from .base import BigQueryBase
from .operations import BigQueryOperations
from .admin import BigQueryAdmin
from .quality import BigQueryQuality
from .schemas import get_table_schemas, get_partitioning_config, get_clustering_fields
from .utils import retry_on_error, validate_table_name, format_table_id

__all__ = [
    # Main interface
    'BigQueryClient',

    # Components
    'BigQueryBase',
    'BigQueryOperations',
    'BigQueryAdmin',
    'BigQueryQuality',

    # Utilities
    'retry_on_error',
    'validate_table_name',
    'format_table_id',

    # Schema functions
    'get_table_schemas',
    'get_partitioning_config',
    'get_clustering_fields',
]

# Version
__version__ = '2.0.0'