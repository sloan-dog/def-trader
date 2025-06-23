"""
Main BigQuery client interface combining all modular components.
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from google.cloud import bigquery

from config.settings import BQ_TABLES
from .admin import BigQueryAdmin
from .quality import BigQueryQuality

from datetime import datetime


class BigQueryClient(BigQueryAdmin, BigQueryQuality):
    """
    Main BigQuery client combining all functionality.

    This class inherits from both BigQueryAdmin and BigQueryQuality,
    which in turn inherit from BigQueryOperations and BigQueryBase,
    providing a complete interface for all BigQuery operations.
    """

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

    def get_feature_data(
            self,
            start_date: str,
            end_date: str,
            tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature data from the feature view.

        Args:
            start_date: Start date
            end_date: End date
            tickers: Optional list of tickers to filter

        Returns:
            DataFrame with feature data
        """
        query = f"""
        SELECT *
        FROM `{self.dataset_ref}.feature_view`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """

        if tickers:
            ticker_list = "','".join(tickers)
            query += f" AND ticker IN ('{ticker_list}')"

        query += " ORDER BY ticker, date"

        return self.query(query)

    def get_latest_predictions(
            self,
            model_version: Optional[str] = None,
            limit: int = 100
    ) -> pd.DataFrame:
        """
        Get latest predictions.

        Args:
            model_version: Optional model version to filter by
            limit: Maximum number of results

        Returns:
            DataFrame with predictions
        """
        query = f"""
        SELECT *
        FROM `{BQ_TABLES['predictions']}`
        WHERE prediction_date = (
            SELECT MAX(prediction_date) 
            FROM `{BQ_TABLES['predictions']}`
        )
        """

        if model_version:
            query += f" AND model_version = '{model_version}'"

        query += f" ORDER BY prediction_date DESC LIMIT {limit}"

        return self.query(query)

    def get_model_performance(
            self,
            model_version: str,
            horizon: str = '7d'
    ) -> Dict[str, float]:
        """
        Get model performance metrics.

        Args:
            model_version: Model version
            horizon: Prediction horizon

        Returns:
            Dictionary with performance metrics
        """
        query = f"""
        WITH predictions_actuals AS (
            SELECT 
                p.ticker,
                p.prediction_date,
                p.horizon_{horizon} as prediction,
                p.confidence_{horizon} as confidence,
                (o2.close - o1.close) / o1.close as actual_return
            FROM `{BQ_TABLES['predictions']}` p
            JOIN `{BQ_TABLES['raw_ohlcv']}` o1
                ON p.ticker = o1.ticker AND p.prediction_date = o1.date
            JOIN `{BQ_TABLES['raw_ohlcv']}` o2
                ON p.ticker = o2.ticker 
                AND o2.date = DATE_ADD(p.prediction_date, INTERVAL {horizon[:-1]} DAY)
            WHERE p.model_version = '{model_version}'
        )
        SELECT 
            COUNT(*) as total_predictions,
            AVG(CASE WHEN SIGN(prediction) = SIGN(actual_return) THEN 1 ELSE 0 END) as direction_accuracy,
            CORR(prediction, actual_return) as correlation,
            AVG(ABS(prediction - actual_return)) as mae,
            SQRT(AVG(POWER(prediction - actual_return, 2))) as rmse
        FROM predictions_actuals
        """

        result = self.query(query)
        return result.iloc[0].to_dict() if not result.empty else {}

    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """
        Clean up old data from tables.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            Dictionary with number of rows deleted per table
        """
        cleanup_results = {}
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)

        tables_to_clean = [
            'raw_ohlcv',
            'technical_indicators',
            'predictions',
            'job_logs'
        ]

        for table in tables_to_clean:
            if self.table_exists(table):
                rows_deleted = self.delete_data(
                    table,
                    f"date < '{cutoff_date.strftime('%Y-%m-%d')}'"
                )
                cleanup_results[table] = rows_deleted

        return cleanup_results

    # Additional methods to add to BigQueryClient for hourly data support

    def get_latest_datetime(self, table_name: str, ticker: str = None) -> Optional[datetime]:
        """
        Get the latest datetime for a ticker in hourly tables.

        Args:
            table_name: Name of the table (without project/dataset prefix)
            ticker: Stock ticker (optional)

        Returns:
            Latest datetime or None if no data exists
        """
        table_ref = BQ_TABLES.get(table_name)
        if not table_ref:
            raise ValueError(f"Unknown table: {table_name}")

        query = f"""
        SELECT MAX(datetime) as latest_datetime
        FROM `{table_ref}`
        """

        if ticker:
            query += f" WHERE ticker = '{ticker}'"

        try:
            result = self.query(query)
            if not result.empty and result['latest_datetime'].iloc[0] is not None:
                return pd.to_datetime(result['latest_datetime'].iloc[0])
            return None
        except Exception as e:
            logger.error(f"Error getting latest datetime: {e}")
            return None


    def create_hourly_tables(self):
        """Create BigQuery tables for hourly data."""
        # Create raw OHLCV hourly table
        raw_ohlcv_hourly_schema = [
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
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                                 default_value_expression="CURRENT_TIMESTAMP()"),
        ]

        # Create technical indicators hourly table
        indicators_hourly_schema = [
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("datetime", "DATETIME", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("hour", "INTEGER", mode="REQUIRED"),
            # Short-term hourly indicators
            bigquery.SchemaField("sma_20h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_140h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_350h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_12h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_84h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_182h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rsi_14h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rsi_98h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE"),
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
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                                 default_value_expression="CURRENT_TIMESTAMP()"),
        ]

        # Create tables with partitioning and clustering
        tables_to_create = [
            ("raw_ohlcv_hourly", raw_ohlcv_hourly_schema),
            ("technical_indicators_hourly", indicators_hourly_schema),
        ]

        for table_name, schema in tables_to_create:
            table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
            table = bigquery.Table(table_id, schema=schema)

            # Set partitioning on date column
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            )

            # Set clustering
            if table_name == "raw_ohlcv_hourly":
                table.clustering_fields = ["ticker", "hour"]
            elif table_name == "technical_indicators_hourly":
                table.clustering_fields = ["ticker", "hour"]

            try:
                table = self.client.create_table(table, exists_ok=True)
                logger.info(f"Created hourly table: {table_id}")
            except Exception as e:
                logger.error(f"Error creating hourly table {table_name}: {e}")


    def create_daily_aggregates_view(self):
        """Create a view that aggregates hourly data to daily."""
        view_id = f"{self.project_id}.{self.dataset_id}.daily_ohlcv_from_hourly"

        view_query = f"""
        CREATE OR REPLACE VIEW `{view_id}` AS
        WITH daily_data AS (
            SELECT 
                ticker,
                date,
                FIRST_VALUE(open) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as open,
                MAX(high) OVER (PARTITION BY ticker, date) as high,
                MIN(low) OVER (PARTITION BY ticker, date) as low,
                LAST_VALUE(close) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as close,
                SUM(volume) OVER (PARTITION BY ticker, date) as volume,
                LAST_VALUE(adjusted_close) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as adjusted_close,
                COUNT(*) OVER (PARTITION BY ticker, date) as hourly_count
            FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
            WHERE hour BETWEEN 9 AND 15  -- Regular trading hours only
        )
        SELECT DISTINCT
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close,
            hourly_count
        FROM daily_data
        WHERE hourly_count >= 6  -- At least 6 hours of data
        """

        try:
            self.client.query(view_query).result()
            logger.info(f"Created daily aggregates view: {view_id}")
        except Exception as e:
            logger.error(f"Error creating daily aggregates view: {e}")


    def get_hourly_data_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get statistics about hourly data coverage."""
        query = f"""
        SELECT 
            ticker,
            COUNT(DISTINCT date) as trading_days,
            COUNT(*) as total_hours,
            AVG(volume) as avg_hourly_volume,
            MIN(datetime) as first_hour,
            MAX(datetime) as last_hour,
            COUNT(DISTINCT hour) as unique_hours,
            SUM(CASE WHEN hour BETWEEN 9 AND 15 THEN 1 ELSE 0 END) as regular_hours,
            SUM(CASE WHEN hour < 9 OR hour > 15 THEN 1 ELSE 0 END) as extended_hours
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY ticker
        ORDER BY ticker
        """

        return self.query(query)