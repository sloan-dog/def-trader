"""
Central configuration settings for the trading signal system - HOURLY VERSION.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Google Cloud settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GCP_ZONE = os.getenv("GCP_ZONE", "us-central1-a")

# BigQuery settings
BQ_DATASET = os.getenv("BQ_DATASET", "trading_signals")
BQ_LOCATION = os.getenv("BQ_LOCATION", "US")
BQ_TABLE_EXPIRATION_DAYS = 365  # For temporary tables

# BigQuery table names - Updated for hourly data
BQ_TABLES = {
    # Legacy daily tables (kept for compatibility)
    "raw_ohlcv": f"{BQ_DATASET}.raw_ohlcv",
    "technical_indicators": f"{BQ_DATASET}.technical_indicators",

    # New hourly tables
    "raw_ohlcv_hourly": f"{BQ_DATASET}.raw_ohlcv_hourly",
    "technical_indicators_hourly": f"{BQ_DATASET}.technical_indicators_hourly",

    # Other tables remain the same
    "macro_indicators": f"{BQ_DATASET}.macro_indicators",
    "sentiment_data": f"{BQ_DATASET}.sentiment_data",
    "temporal_features": f"{BQ_DATASET}.temporal_features",
    "stock_metadata": f"{BQ_DATASET}.stock_metadata",
    "predictions": f"{BQ_DATASET}.predictions",
    "model_metadata": f"{BQ_DATASET}.model_metadata",
    "job_logs": f"{BQ_DATASET}.job_logs",
    "daily_aggregates": f"{BQ_DATASET}.daily_aggregates",  # Daily aggregates from hourly
}

# Alpha Vantage settings - Updated for premium account
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_RATE_LIMIT = 75  # Premium tier: 75 calls per minute
ALPHA_VANTAGE_TIMEOUT = 30  # seconds
ALPHA_VANTAGE_CALL_DELAY = float(os.getenv("ALPHA_VANTAGE_CALL_DELAY", "2.0"))  # Enforced delay between calls in seconds

# Model settings - Updated for hourly data
MODEL_CONFIG = {
    "prediction_horizons": [1, 24, 168, 720],  # hours (1hr, 1d, 1w, 1m)
    "historical_window": 2160,  # hours (~90 days)
    "batch_size": 64,  # Larger batches for more data
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "model_checkpoint_dir": str(MODELS_DIR / "checkpoints"),
    "model_registry_bucket": f"{GCP_PROJECT_ID}-model-registry",
    "data_frequency": "hourly",
}

# GNN architecture settings - Updated for hourly patterns
GNN_CONFIG = {
    "node_features_dim": 128,  # More features for hourly data
    "edge_features_dim": 32,
    "hidden_dim": 256,  # Larger model for more complex patterns
    "num_gnn_layers": 4,
    "num_temporal_layers": 3,
    "dropout_rate": 0.2,
    "attention_heads": 8,
    "hourly_embeddings": True,  # Add hour-of-day embeddings
}

# Vertex AI settings
VERTEX_AI_CONFIG = {
    "location": GCP_REGION,
    "project": GCP_PROJECT_ID,
    "experiment": "temporal-gnn-trading-hourly",
    "experiment_description": "GNN-based hourly trading signal predictions with multi-horizon forecasting",
    "metadata_store": f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/metadataStores/trading-signals-metadata",
    "tensorboard": f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/tensorboards/trading-signals-experiments",
    "staging_bucket": f"gs://{GCP_PROJECT_ID}-vertex-staging",
}

# Technical indicators settings - Updated for hourly data
TECHNICAL_INDICATORS = {
    # Short-term hourly indicators
    "rsi": {"periods": [14, 98]},  # 14-hour and ~14-day equivalent
    "ema": {"periods": [12, 84, 182]},  # 12-hour, ~12-day, ~26-day
    "sma": {"periods": [20, 140, 350]},  # 20-hour, ~20-day, ~50-day
    "macd": {"fast": 84, "slow": 182, "signal": 63},  # Hourly equivalents
    "bollinger_bands": {"periods": [20, 140], "std_dev": 2},
    "atr": {"periods": [14, 98]},
    "adx": {"period": 14},
    "obv": {},
    "vwap": {"reset": "daily"},  # Reset VWAP daily for hourly data

    # Hourly-specific indicators
    "hourly_patterns": {
        "market_hours": [9, 10, 11, 12, 13, 14, 15],  # Regular trading hours
        "volume_profile": True,
        "price_levels": True,
    }
}

# Data ingestion settings - Updated for hourly
INGESTION_CONFIG = {
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    "chunk_size": 10000,  # Larger chunks for hourly data
    "parallel_workers": 10,  # More workers with higher rate limit
    "backfill_start_date": "2024-01-01",
    "data_frequency": "hourly",
    "hours_per_day": 13,  # Including extended hours
    "regular_hours_per_day": 7,
    "extended_hours": True,
}

# Feature engineering settings - Updated for hourly
FEATURE_CONFIG = {
    "correlation_threshold": 0.95,
    "missing_data_threshold": 0.3,
    "scaling_method": "robust",
    "graph_construction": {
        "sector_weight": 0.3,
        "correlation_weight": 0.4,
        "market_cap_weight": 0.3,
        "min_correlation": 0.3,
        "correlation_window": 720,  # hours (~30 days)
    },
    "hourly_features": {
        "hour_of_day": True,
        "is_market_open": True,
        "is_first_hour": True,
        "is_last_hour": True,
        "volume_profile": True,
        "intraday_momentum": True,
    }
}

# Validation settings - Updated for hourly
VALIDATION_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "walk_forward_windows": 12,
    "min_hours_per_ticker": 1000,  # Minimum hourly data for validation
}

# Cloud Run job settings - Updated for larger data processing
CLOUD_RUN_CONFIG = {
    "cpu": "4",  # More CPU for hourly data
    "memory": "8Gi",  # More memory
    "max_instances": 20,
    "timeout": 3600,
    "service_account": f"trading-system@{GCP_PROJECT_ID}.iam.gserviceaccount.com",
}

# Vertex AI training settings - Updated for larger models
VERTEX_AI_TRAINING_CONFIG = {
    "machine_type": "n1-highmem-16",  # More memory for hourly data
    "accelerator_type": "NVIDIA_TESLA_V100",  # Better GPU
    "accelerator_count": 2,
    "boot_disk_size_gb": 200,
    "training_image": f"gcr.io/{GCP_PROJECT_ID}/trading-signal-trainer:latest",
}

# Logging settings
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "rotation": "100 MB",
    "retention": "30 days",
    "compression": "gz",
}

# Alert settings
ALERT_CONFIG = {
    "email_recipients": os.getenv("ALERT_EMAILS", "").split(","),
    "slack_webhook": os.getenv("SLACK_WEBHOOK", ""),
    "pagerduty_key": os.getenv("PAGERDUTY_KEY", ""),
}

# Macro indicators list
MACRO_INDICATORS = [
    "gdp", "cpi", "pce", "nfp", "unemployment_rate",
    "fed_funds_rate", "yield_curve_spread", "retail_sales",
    "ism_manufacturing", "ism_services", "consumer_confidence",
    "wti_crude", "brent_crude", "china_pmi", "china_gdp",
    "m2_money_supply"
]

# Load stock configuration
def load_stocks_config() -> Dict[str, List[str]]:
    """Load stock tickers configuration from YAML file."""
    config_path = CONFIG_DIR / "stocks_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load indicator configuration
def load_indicators_config() -> Dict:
    """Load technical indicators configuration from YAML file."""
    config_path = CONFIG_DIR / "indicators_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)