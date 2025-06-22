"""
BigQuery data quality and validation operations.
"""
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from config.settings import BQ_TABLES
from .operations import BigQueryOperations


class BigQueryQuality(BigQueryOperations):
    """Data quality and validation operations for BigQuery."""

    def check_data_quality(
            self,
            table_name: str,
            date_range: Optional[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run data quality checks on a table.

        Args:
            table_name: Table to check
            date_range: Optional date range tuple (start, end)

        Returns:
            Dictionary with quality check results
        """
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
            checks['duplicate_count'] = self._check_duplicates(table_name, date_range)

        # Check for nulls in critical columns
        checks['null_counts'] = self._check_nulls(table_name, date_range)

        # Check data freshness
        checks['freshness'] = self._check_freshness(table_name)

        return checks

    def _check_duplicates(
            self,
            table_name: str,
            date_range: Optional[Tuple[str, str]] = None
    ) -> int:
        """Check for duplicate records."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT COUNT(*) as duplicate_count
        FROM (
            SELECT ticker, date, COUNT(*) as cnt
            FROM `{table_id}`
            {f"WHERE date BETWEEN '{date_range[0]}' AND '{date_range[1]}'" if date_range else ""}
            GROUP BY ticker, date
            HAVING cnt > 1
        )
        """

        result = self.query(query)
        return result['duplicate_count'].iloc[0]

    def _check_nulls(
            self,
            table_name: str,
            date_range: Optional[Tuple[str, str]] = None
    ) -> Dict[str, int]:
        """Check for null values in critical columns."""
        critical_columns = {
            'raw_ohlcv': ['ticker', 'date', 'close', 'volume'],
            'technical_indicators': ['ticker', 'date', 'rsi', 'ema_20'],
            'macro_indicators': ['date', 'gdp', 'cpi'],
        }

        null_counts = {}

        if table_name in critical_columns:
            table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

            for col in critical_columns[table_name]:
                query = f"""
                SELECT COUNT(*) as null_count
                FROM `{table_id}`
                WHERE {col} IS NULL
                """
                if date_range:
                    query += f" AND date BETWEEN '{date_range[0]}' AND '{date_range[1]}'"

                result = self.query(query)
                null_counts[col] = result['null_count'].iloc[0]

        return null_counts

    def _check_freshness(self, table_name: str) -> Dict[str, Any]:
        """Check data freshness."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT 
            MAX(date) as latest_date,
            MIN(date) as earliest_date,
            COUNT(DISTINCT date) as unique_dates
        FROM `{table_id}`
        WHERE date IS NOT NULL
        """

        result = self.query(query)

        if result.empty:
            return {'error': 'No data found'}

        return {
            'latest_date': str(result['latest_date'].iloc[0]),
            'earliest_date': str(result['earliest_date'].iloc[0]),
            'unique_dates': result['unique_dates'].iloc[0],
        }

    def validate_data_integrity(
            self,
            table_name: str,
            validation_rules: Optional[Dict[str, str]] = None
    ) -> Dict[str, bool]:
        """
        Validate data integrity with custom rules.

        Args:
            table_name: Table to validate
            validation_rules: Dictionary of rule_name -> SQL condition

        Returns:
            Dictionary of rule_name -> passed (bool)
        """
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        # Default validation rules
        default_rules = {
            'raw_ohlcv': {
                'price_consistency': 'high >= low AND high >= open AND high >= close',
                'positive_volume': 'volume >= 0',
                'positive_prices': 'open > 0 AND high > 0 AND low > 0 AND close > 0',
            },
            'technical_indicators': {
                'rsi_range': 'rsi IS NULL OR (rsi >= 0 AND rsi <= 100)',
                'valid_ema': 'ema_20 IS NULL OR ema_20 > 0',
            }
        }

        rules = validation_rules or default_rules.get(table_name, {})
        results = {}

        for rule_name, condition in rules.items():
            query = f"""
            SELECT COUNT(*) as violation_count
            FROM `{table_id}`
            WHERE NOT ({condition})
            """

            result = self.query(query)
            violation_count = result['violation_count'].iloc[0]
            results[rule_name] = violation_count == 0

            if violation_count > 0:
                logger.warning(f"Rule '{rule_name}' failed with {violation_count} violations")

        return results

    def generate_table_hash(self, table_name: str) -> str:
        """
        Generate hash of table content for version tracking.

        Args:
            table_name: Table name

        Returns:
            Hash string
        """
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
        return str(result['table_hash'].iloc[0]) if not result.empty else "0"

    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a table.

        Args:
            table_name: Table name

        Returns:
            Dictionary with table statistics
        """
        table_ref = self.get_table_reference(table_name)
        table = self.client.get_table(table_ref)

        stats = {
            'basic_info': {
                'num_rows': table.num_rows,
                'size_mb': table.num_bytes / 1024 / 1024 if table.num_bytes else 0,
                'created': str(table.created),
                'modified': str(table.modified),
            }
        }

        # Get column statistics for numeric columns
        if table_name == 'raw_ohlcv':
            stats['price_stats'] = self._get_price_statistics(table_name)

        return stats

    def _get_price_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get price statistics for OHLCV data."""
        table_id = BQ_TABLES.get(table_name, f"{self.dataset_ref}.{table_name}")

        query = f"""
        SELECT 
            AVG(close) as avg_price,
            MIN(close) as min_price,
            MAX(close) as max_price,
            STDDEV(close) as price_stddev,
            AVG(volume) as avg_volume,
            MAX(volume) as max_volume
        FROM `{table_id}`
        WHERE close > 0
        """

        result = self.query(query)
        return result.iloc[0].to_dict() if not result.empty else {}