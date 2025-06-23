"""
BigQuery table schema definitions - Updated for HOURLY data.
"""
from typing import Dict, List
from google.cloud import bigquery


def get_table_schemas() -> Dict[str, List[bigquery.SchemaField]]:
    """Get all table schemas for the trading signal system."""
    return {
        # Daily tables (kept for backward compatibility)
        'raw_ohlcv': _get_ohlcv_schema(),
        'technical_indicators': _get_technical_indicators_schema(),

        # NEW HOURLY TABLES
        'raw_ohlcv_hourly': _get_ohlcv_hourly_schema(),
        'technical_indicators_hourly': _get_technical_indicators_hourly_schema(),

        # Other tables remain unchanged
        'macro_indicators': _get_macro_indicators_schema(),
        'sentiment_data': _get_sentiment_data_schema(),
        'temporal_features': _get_temporal_features_schema(),
        'stock_metadata': _get_stock_metadata_schema(),
        'predictions': _get_predictions_schema(),
        'model_metadata': _get_model_metadata_schema(),
        'job_logs': _get_job_logs_schema(),
    }


def _get_ohlcv_schema() -> List[bigquery.SchemaField]:
    """Schema for raw OHLCV DAILY data (legacy)."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("volume", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("adjusted_close", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_ohlcv_hourly_schema() -> List[bigquery.SchemaField]:
    """Schema for raw OHLCV HOURLY data."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("datetime", "DATETIME", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("hour", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("volume", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("adjusted_close", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_technical_indicators_schema() -> List[bigquery.SchemaField]:
    """Schema for technical indicators DAILY (legacy)."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("rsi", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ema_9", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ema_20", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ema_50", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("vwap", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("macd_hist", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bb_upper", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bb_middle", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bb_lower", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("atr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("sma_20", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("sma_50", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("sma_200", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("adx", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("obv", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_technical_indicators_hourly_schema() -> List[bigquery.SchemaField]:
    """Schema for technical indicators HOURLY."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("datetime", "DATETIME", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("hour", "INTEGER", mode="REQUIRED"),

        # Short-term hourly indicators
        bigquery.SchemaField("sma_20h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("sma_140h", "FLOAT64", mode="NULLABLE"),  # ~20 day equivalent
        bigquery.SchemaField("sma_350h", "FLOAT64", mode="NULLABLE"),  # ~50 day equivalent
        bigquery.SchemaField("ema_12h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ema_84h", "FLOAT64", mode="NULLABLE"),   # ~12 day equivalent
        bigquery.SchemaField("ema_182h", "FLOAT64", mode="NULLABLE"),  # ~26 day equivalent
        bigquery.SchemaField("rsi_14h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("rsi_98h", "FLOAT64", mode="NULLABLE"),   # ~14 day equivalent

        # MACD
        bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE"),

        # Bollinger Bands
        bigquery.SchemaField("bb_upper_20h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bb_middle_20h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("bb_lower_20h", "FLOAT64", mode="NULLABLE"),

        # Volume indicators
        bigquery.SchemaField("volume_sma_20h", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("volume_ratio", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("obv", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("vwap", "FLOAT64", mode="NULLABLE"),

        # Hourly-specific indicators
        bigquery.SchemaField("hour_of_day", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("is_market_open", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("is_first_hour", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("is_last_hour", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("high_of_day", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("low_of_day", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("pct_from_high", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("pct_from_low", "FLOAT64", mode="NULLABLE"),

        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_macro_indicators_schema() -> List[bigquery.SchemaField]:
    """Schema for macro economic indicators."""
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("indicator", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("value", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("yoy_change", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_sentiment_data_schema() -> List[bigquery.SchemaField]:
    """Schema for sentiment data."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("news_sentiment", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("social_sentiment", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("analyst_rating", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("composite_sentiment", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_temporal_features_schema() -> List[bigquery.SchemaField]:
    """Schema for temporal features."""
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("is_holiday", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("is_earnings_season", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("is_fomc_week", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("days_to_quarter_end", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("market_regime", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_stock_metadata_schema() -> List[bigquery.SchemaField]:
    """Schema for stock metadata."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("industry", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("market_cap_category", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("exchange", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_predictions_schema() -> List[bigquery.SchemaField]:
    """Schema for model predictions."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("prediction_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("horizon_days", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("predicted_return", "FLOAT64", mode="REQUIRED"),
        bigquery.SchemaField("confidence_score", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("predicted_direction", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("feature_importance", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_model_metadata_schema() -> List[bigquery.SchemaField]:
    """Schema for model metadata."""
    return [
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("training_start_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("training_end_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("validation_metrics", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("hyperparameters", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("feature_importance", "JSON", mode="NULLABLE"),
        bigquery.SchemaField("model_path", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_job_logs_schema() -> List[bigquery.SchemaField]:
    """Schema for job logs."""
    return [
        bigquery.SchemaField("job_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("run_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("end_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("duration_hours", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("success", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("parameters", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("step_results", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
    ]


def get_partitioning_config(table_name: str) -> Dict[str, str]:
    """Get partitioning configuration for a table."""
    partitioned_tables = {
        'raw_ohlcv': 'date',
        'raw_ohlcv_hourly': 'date',  # Partition hourly data by date
        'technical_indicators': 'date',
        'technical_indicators_hourly': 'date',  # Partition hourly indicators by date
        'sentiment_data': 'date',
        'predictions': 'prediction_date',
        'temporal_features': 'date',
        'job_logs': 'run_date',
    }

    if table_name in partitioned_tables:
        return {
            'type': 'DAY',
            'field': partitioned_tables[table_name]
        }
    return {}


def get_clustering_fields(table_name: str) -> List[str]:
    """Get clustering fields for a table."""
    clustering_config = {
        'raw_ohlcv': ['ticker'],
        'raw_ohlcv_hourly': ['ticker', 'hour'],  # Cluster by ticker and hour
        'technical_indicators': ['ticker'],
        'technical_indicators_hourly': ['ticker', 'hour'],  # Cluster by ticker and hour
        'sentiment_data': ['ticker', 'sector'],
        'predictions': ['ticker', 'model_version'],
    }
    return clustering_config.get(table_name, [])