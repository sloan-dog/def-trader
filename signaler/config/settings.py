"""
Central configuration settings for the trading signal system.
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

# BigQuery table names
BQ_TABLES = {
    "raw_ohlcv": f"{BQ_DATASET}.raw_ohlcv",
    "technical_indicators": f"{BQ_DATASET}.technical_indicators",
    "macro_indicators": f"{BQ_DATASET}.macro_indicators",
    "sentiment_data": f"{BQ_DATASET}.sentiment_data",
    "temporal_features": f"{BQ_DATASET}.temporal_features",
    "stock_metadata": f"{BQ_DATASET}.stock_metadata",
    "predictions": f"{BQ_DATASET}.predictions",
    "model_metadata": f"{BQ_DATASET}.model_metadata",
}

# Alpha Vantage settings
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_RATE_LIMIT = 5  # calls per minute for free tier
ALPHA_VANTAGE_TIMEOUT = 30  # seconds

# Model settings
MODEL_CONFIG = {
    "prediction_horizons": [1, 7, 30, 60],  # days
    "historical_window": 90,  # days
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "early_stopping_patience": 10,
    "model_checkpoint_dir": str(MODELS_DIR / "checkpoints"),
    "model_registry_bucket": f"{GCP_PROJECT_ID}-model-registry",
}

# GNN architecture settings
GNN_CONFIG = {
    "node_features_dim": 64,
    "edge_features_dim": 16,
    "hidden_dim": 128,
    "num_gnn_layers": 3,
    "num_temporal_layers": 2,
    "dropout_rate": 0.2,
    "attention_heads": 4,
}

# Vertex AI settings
VERTEX_AI_CONFIG = {
    "location": GCP_REGION,
    "project": GCP_PROJECT_ID,
    "experiment": "temporal-gnn-trading",
    "experiment_description": "GNN-based trading signal predictions with multi-horizon forecasting",
    "metadata_store": f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/metadataStores/trading-signals-metadata",
    "tensorboard": f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/tensorboards/trading-signals-experiments",
    "staging_bucket": f"gs://{GCP_PROJECT_ID}-vertex-staging",
}

# Technical indicators settings
TECHNICAL_INDICATORS = {
    "rsi": {"period": 14},
    "ema": {"periods": [9, 20, 50]},
    "sma": {"periods": [20, 50, 200]},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_bands": {"period": 20, "std_dev": 2},
    "atr": {"period": 14},
    "adx": {"period": 14},
    "obv": {},
    "vwap": {},
}

# Data ingestion settings
INGESTION_CONFIG = {
    "max_retries": 3,
    "retry_delay": 5,  # seconds
    "chunk_size": 1000,  # rows per BigQuery insert
    "parallel_workers": 4,
    "backfill_start_date": "2020-01-01",
}

# Logging settings
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "rotation": "100 MB",
    "retention": "30 days",
    "compression": "gz",
}

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

# Macro indicators list
MACRO_INDICATORS = [
    "gdp", "cpi", "pce", "nfp", "unemployment_rate",
    "fed_funds_rate", "yield_curve_spread", "retail_sales",
    "ism_manufacturing", "ism_services", "consumer_confidence",
    "wti_crude", "brent_crude", "china_pmi", "china_gdp",
    "m2_money_supply"
]

# Cloud Run job settings
CLOUD_RUN_CONFIG = {
    "cpu": "2",
    "memory": "4Gi",
    "max_instances": 10,
    "timeout": 3600,  # 1 hour
    "service_account": f"trading-system@{GCP_PROJECT_ID}.iam.gserviceaccount.com",
}

# Vertex AI training settings
VERTEX_AI_TRAINING_CONFIG = {
    "machine_type": "n1-standard-8",
    "accelerator_type": "NVIDIA_TESLA_T4",
    "accelerator_count": 1,
    "boot_disk_size_gb": 100,
    "training_image": f"gcr.io/{GCP_PROJECT_ID}/trading-signal-trainer:latest",
}

# Feature engineering settings
FEATURE_CONFIG = {
    "correlation_threshold": 0.95,  # Remove highly correlated features
    "missing_data_threshold": 0.3,  # Remove features with >30% missing
    "scaling_method": "robust",  # robust, standard, or minmax
    "graph_construction": {
        "sector_weight": 0.3,
        "correlation_weight": 0.4,
        "market_cap_weight": 0.3,
        "min_correlation": 0.3,  # Minimum correlation for edge creation
    }
}

# Validation settings
VALIDATION_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "walk_forward_windows": 12,  # For time series cross-validation
}

# Alert settings
ALERT_CONFIG = {
    "email_recipients": os.getenv("ALERT_EMAILS", "").split(","),
    "slack_webhook": os.getenv("SLACK_WEBHOOK", ""),
    "pagerduty_key": os.getenv("PAGERDUTY_KEY", ""),
}