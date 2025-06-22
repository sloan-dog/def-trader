"""
Main BigQuery client interface combining all modular components.
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from google.cloud import bigquery

from config.settings import BQ_TABLES
from .admin import BigQueryAdmin
from .quality import BigQueryQuality


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