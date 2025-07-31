"""
Storage configuration for Parquet/GCS setup.
"""
import os
from typing import Dict, List

# GCS Configuration
GCS_BUCKET = os.getenv('GCS_BUCKET', 'def-trader-data')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')

# Storage paths
STORAGE_PATHS = {
    'ohlcv': 'market_data/ohlcv',
    'technical_indicators': 'market_data/indicators',
    'sentiment': 'alternative_data/sentiment',
    'macro': 'economic_data/macro',
    'news': 'alternative_data/news',
    'options': 'market_data/options',
    'features': 'ml/features',
    'graphs': 'ml/graphs',
    'models': 'ml/models',
    'predictions': 'ml/predictions'
}

# Partition strategies
PARTITION_SCHEMES = {
    'ohlcv': ['symbol', 'year', 'month', 'day'],
    'technical_indicators': ['symbol', 'date'],
    'sentiment': ['symbol', 'date'],
    'macro': ['date'],
    'news': ['date', 'hour'],
    'options': ['symbol', 'date'],
    'features': ['date'],
    'graphs': ['date'],
    'predictions': ['model_version', 'date']
}

# Compression settings
COMPRESSION_CONFIG = {
    'default': 'snappy',  # Fast compression/decompression
    'archival': 'zstd',   # Better compression for long-term storage
    'real_time': 'lz4'    # Fastest for real-time data
}

# Data retention policies (days)
RETENTION_POLICIES = {
    'ohlcv': -1,  # Keep forever
    'technical_indicators': 365,
    'sentiment': 180,
    'macro': -1,  # Keep forever
    'news': 90,
    'options': 365,
    'features': 30,
    'graphs': 30,
    'predictions': 90
}

# Schema definitions for validation
SCHEMAS = {
    'ohlcv': {
        'required_columns': ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
        'dtypes': {
            'symbol': 'string',
            'timestamp': 'datetime64[ns]',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64'
        }
    },
    'sentiment': {
        'required_columns': ['symbol', 'timestamp', 'source', 'sentiment_score'],
        'dtypes': {
            'symbol': 'string',
            'timestamp': 'datetime64[ns]',
            'source': 'string',
            'sentiment_score': 'float64'
        }
    },
    'macro': {
        'required_columns': ['timestamp', 'indicator', 'value'],
        'dtypes': {
            'timestamp': 'datetime64[ns]',
            'indicator': 'string',
            'value': 'float64'
        }
    }
}

# Performance settings
PERFORMANCE_CONFIG = {
    'max_parallel_writes': 4,
    'chunk_size': 100000,  # Rows per chunk
    'cache_ttl_seconds': 3600,  # 1 hour
    'max_memory_mb': 2048
}

def get_storage_path(data_type: str) -> str:
    """Get the storage path for a data type."""
    return STORAGE_PATHS.get(data_type, f'raw/{data_type}')

def get_partition_cols(data_type: str) -> List[str]:
    """Get partition columns for a data type."""
    return PARTITION_SCHEMES.get(data_type, ['date'])

def get_schema(data_type: str) -> Dict:
    """Get schema definition for a data type."""
    return SCHEMAS.get(data_type, {})