"""
BigQuery client utilities for data operations.
"""
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import time
from loguru import logger
from functools import wraps
import hashlib

from config.settings import (
    GCP_PROJECT_ID,
    BQ_DATASET,
    BQ_LOCATION,
    BQ_TABLES,
    INGESTION_CONFIG
)


def retry_on_error(max_retries: int = 3, delay: int = 5):
    """Decorator to retry BigQuery operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except GoogleCloudError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


class BigQueryClient:
    """Wrapper for BigQuery operations with error handling and utilities."""

    def __init__(self, project_id: str = GCP_PROJECT_ID):
        """Initialize BigQuery client."""
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = BQ_DATASET
        self.dataset_ref = f"{project_id}.{BQ_DATASET}"

    def create_dataset_if_not_exists(self) -> None:
        """Create BigQuery dataset if it doesn't exist."""
        dataset_id = f"{self.project_id}.{self.dataset_id}"

        try:
            self.client.get_dataset(dataset_id)
            logger.info(f"Dataset {dataset_id} already exists")
        except Exception:
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = BQ_LOCATION
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id}")

    def create_tables(self) -> None:
        """Create all required tables with schemas."""
        schemas = self._get_table_schemas()

        for table_name, schema in schemas.items():
            table_id = f"{self.project_id}.{BQ_DATASET}.{table_name}"

            try:
                self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists")
            except Exception:
                table = bigquery.Table(table_id, schema=schema)

                # Add partitioning for time-series tables
                if table_name in ['raw_ohlcv', 'technical_indicators', 'sentiment_data']:
                    table.time_partitioning = bigquery.TimePartitioning(
                        type_=bigquery.TimePartitioningType.DAY,
                        field="date"
                    )

                table = self.client.create_table(table)
                logger.info(f"Created table {table_id}")

    @retry_on_error()
    def insert_dataframe(
            self,
            df: pd.DataFrame,
            table_name: str,
            chunk_size: int = None,
            if_exists: str = 'append'
    ) -> None:
        """Insert DataFrame to BigQuery table."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")
        chunk_size = chunk_size or INGESTION_CONFIG['chunk_size']

        # Add insertion timestamp
        df['inserted_at'] = pd.Timestamp.now()

        # Insert in chunks
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]

            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND
                if if_exists == 'append'
                else bigquery.WriteDisposition.WRITE_TRUNCATE
            )

            job = self.client.load_table_from_dataframe(
                chunk, table_id, job_config=job_config
            )
            job.result()  # Wait for job to complete

            logger.info(f"Inserted {len(chunk)} rows to {table_id}")

    @retry_on_error()
    def query(
            self,
            query: str,
            params: Optional[List[bigquery.ScalarQueryParameter]] = None
    ) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        job_config = bigquery.QueryJobConfig(
            query_parameters=params or [],
            use_query_cache=True
        )

        query_job = self.client.query(query, job_config=job_config)
        results = query_job.result()

        return results.to_dataframe()

    def get_latest_date(self, table_name: str, ticker: str = None) -> Optional[pd.Timestamp]:
        """Get the latest date for a ticker in a table."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT MAX(date) as latest_date
        FROM `{table_id}`
        """

        if ticker:
            query += f" WHERE ticker = '{ticker}'"

        try:
            result = self.query(query)
            if not result.empty and result['latest_date'].iloc[0]:
                return pd.Timestamp(result['latest_date'].iloc[0])
        except Exception as e:
            logger.warning(f"Could not get latest date: {e}")

        return None

    def check_data_quality(self, table_name: str, date_range: tuple = None) -> Dict[str, Any]:
        """Run data quality checks on a table."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        checks = {}

        # Check row count
        query = f"SELECT COUNT(*) as row_count FROM `{table_id}`"
        if date_range:
            query += f" WHERE date BETWEEN '{date_range[0]}' AND '{date_range[1]}'"

        result = self.query(query)
        checks['row_count'] = result['row_count'].iloc[0]

        # Check for duplicates
        if table_name in ['raw_ohlcv', 'technical_indicators']:
            query = f"""
            SELECT COUNT(*) as duplicate_count
            FROM (
                SELECT ticker, date, COUNT(*) as cnt
                FROM `{table_id}`
                GROUP BY ticker, date
                HAVING cnt > 1
            )
            """
            result = self.query(query)
            checks['duplicate_count'] = result['duplicate_count'].iloc[0]

        # Check for nulls in critical columns
        critical_columns = {
            'raw_ohlcv': ['ticker', 'date', 'close', 'volume'],
            'technical_indicators': ['ticker', 'date', 'rsi', 'ema_20'],
            'macro_indicators': ['date', 'gdp', 'cpi'],
        }

        if table_name in critical_columns:
            null_checks = {}
            for col in critical_columns[table_name]:
                query = f"""
                SELECT COUNT(*) as null_count
                FROM `{table_id}`
                WHERE {col} IS NULL
                """
                if date_range:
                    query += f" AND date BETWEEN '{date_range[0]}' AND '{date_range[1]}'"

                result = self.query(query)
                null_checks[col] = result['null_count'].iloc[0]

            checks['null_counts'] = null_checks

        return checks

    def create_feature_view(self) -> None:
        """Create a materialized view for feature engineering."""
        view_query = f"""
        CREATE OR REPLACE MATERIALIZED VIEW `{self.dataset_ref}.feature_view`
        PARTITION BY DATE(date)
        AS
        SELECT 
            o.ticker,
            o.date,
            o.open,
            o.high,
            o.low,
            o.close,
            o.volume,
            o.adjusted_close,
            t.rsi,
            t.ema_9,
            t.ema_20,
            t.ema_50,
            t.vwap,
            t.macd,
            t.macd_signal,
            t.bb_upper,
            t.bb_middle,
            t.bb_lower,
            t.atr,
            t.sma_20,
            t.sma_50,
            t.sma_200,
            t.adx,
            t.obv,
            s.sentiment_score,
            s.volume_mentions,
            tf.day_of_week,
            tf.month,
            tf.quarter,
            tf.is_holiday,
            tf.days_to_next_holiday,
            tf.is_earnings_season,
            sm.sector,
            sm.industry,
            sm.market_cap_category
        FROM `{BQ_TABLES['raw_ohlcv']}` o
        LEFT JOIN `{BQ_TABLES['technical_indicators']}` t
            ON o.ticker = t.ticker AND o.date = t.date
        LEFT JOIN `{BQ_TABLES['sentiment_data']}` s
            ON o.ticker = s.ticker AND o.date = s.date
        LEFT JOIN `{BQ_TABLES['temporal_features']}` tf
            ON o.date = tf.date
        LEFT JOIN `{BQ_TABLES['stock_metadata']}` sm
            ON o.ticker = sm.ticker
        WHERE o.date >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR)
        """

        self.client.query(view_query).result()
        logger.info("Created feature view")

    def _get_table_schemas(self) -> Dict[str, List[bigquery.SchemaField]]:
        """Define schemas for all tables."""
        return {
            'raw_ohlcv': [
                bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
                bigquery.SchemaField("volume", "INT64", mode="REQUIRED"),
                bigquery.SchemaField("adjusted_close", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            'technical_indicators': [
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
            ],
            'macro_indicators': [
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
            ],
            'sentiment_data': [
                bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("sector", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("sentiment_score", "FLOAT64", mode="NULLABLE"),
                bigquery.SchemaField("volume_mentions", "INT64", mode="NULLABLE"),
                bigquery.SchemaField("source", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("inserted_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            'temporal_features': [
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
            ],
            'stock_metadata': [
                bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("sector", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("industry", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("market_cap_category", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("exchange", "STRING", mode="NULLABLE"),
                bigquery.SchemaField("updated_at", "TIMESTAMP", mode="REQUIRED"),
            ],
            'predictions': [
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
            ],
            'model_metadata': [
                bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("model_type", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("training_start_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("training_end_date", "DATE", mode="REQUIRED"),
                bigquery.SchemaField("validation_metrics", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("hyperparameters", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("feature_importance", "JSON", mode="NULLABLE"),
                bigquery.SchemaField("model_path", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            ],
        }

    def generate_table_hash(self, table_name: str) -> str:
        """Generate hash of table content for version tracking."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT 
            FARM_FINGERPRINT(
                STRING_AGG(
                    CAST(TO_JSON_STRING(t) AS STRING),
                    '' ORDER BY t
                )
            ) as table_hash
        FROM `{table_id}` t
        """

        result = self.query(query)
        return str(result['table_hash'].iloc[0]) if not result.empty else None