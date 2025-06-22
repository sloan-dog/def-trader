"""
BigQuery table schema definitions.
"""
from typing import Dict, List
from google.cloud import bigquery


def get_table_schemas() -> Dict[str, List[bigquery.SchemaField]]:
    """Get all table schemas for the trading signal system."""
    return {
        'raw_ohlcv': _get_ohlcv_schema(),
        'technical_indicators': _get_technical_indicators_schema(),
        'macro_indicators': _get_macro_indicators_schema(),
        'sentiment_data': _get_sentiment_data_schema(),
        'temporal_features': _get_temporal_features_schema(),
        'stock_metadata': _get_stock_metadata_schema(),
        'predictions': _get_predictions_schema(),
        'model_metadata': _get_model_metadata_schema(),
        'job_logs': _get_job_logs_schema(),
    }


def _get_ohlcv_schema() -> List[bigquery.SchemaField]:
    """Schema for raw OHLCV data."""
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


def _get_technical_indicators_schema() -> List[bigquery.SchemaField]:
    """Schema for technical indicators."""
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


def _get_macro_indicators_schema() -> List[bigquery.SchemaField]:
    """Schema for macro economic indicators."""
    fields = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("gdp", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("gdp_growth", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("cpi", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("cpi_yoy", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("pce", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("nfp", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("unemployment_rate", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("fed_funds_rate", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("yield_curve_spread", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("retail_sales", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ism_manufacturing", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ism_services", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("consumer_confidence", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("wti_crude", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("brent_crude", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("china_pmi", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("china_gdp", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("m2_money_supply", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    return fields


def _get_sentiment_data_schema() -> List[bigquery.SchemaField]:
    """Schema for sentiment data."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("sentiment_score", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("volume_mentions", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_temporal_features_schema() -> List[bigquery.SchemaField]:
    """Schema for temporal features."""
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("day_of_week", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("month", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("quarter", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("year", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("is_holiday", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("days_to_next_holiday", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("is_earnings_season", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("is_month_start", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("is_month_end", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("is_quarter_start", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("is_quarter_end", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_stock_metadata_schema() -> List[bigquery.SchemaField]:
    """Schema for stock metadata."""
    return [
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("sector", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("industry", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("market_cap_category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("exchange", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def _get_predictions_schema() -> List[bigquery.SchemaField]:
    """Schema for predictions."""
    return [
        bigquery.SchemaField("prediction_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("prediction_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("horizon_1d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("horizon_7d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("horizon_30d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("horizon_60d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("confidence_1d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("confidence_7d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("confidence_30d", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("confidence_60d", "FLOAT64", mode="NULLABLE"),
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
        'technical_indicators': 'date',
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
        'technical_indicators': ['ticker'],
        'sentiment_data': ['ticker', 'sector'],
        'predictions': ['ticker', 'model_version'],
    }
    return clustering_config.get(table_name, [])