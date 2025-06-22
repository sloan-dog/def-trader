"""
Backfill historical data for all data sources.
"""
import click
import sys
from datetime import datetime, timedelta
from loguru import logger
import time

from src.data_ingestion.ohlcv_fetcher import OHLCVFetcher
from src.data_ingestion.macro_data_fetcher import MacroDataFetcher
from src.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.utils.bigquery import BigQueryClient
from config.settings import load_stocks_config, BQ_TABLES
from typing import Dict
import pandas as pd


class BackfillJob:
    """Orchestrate historical data backfill."""

    def __init__(self):
        """Initialize backfill components."""
        self.ohlcv_fetcher = OHLCVFetcher()
        self.macro_fetcher = MacroDataFetcher()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.bq_client = BigQueryClient()
        self.stocks_config = load_stocks_config()

        # Configure logging
        logger.add(
            "logs/backfill_{time}.log",
            rotation="100 MB",
            retention="7 days",
            level="INFO"
        )

    def run(
            self,
            start_date: str,
            end_date: str = None,
            data_types: str = 'all',
            batch_size: int = 10
    ):
        """Run backfill job."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Starting backfill from {start_date} to {end_date}")
        logger.info(f"Data types: {data_types}")

        results = {
            'start_date': start_date,
            'end_date': end_date,
            'start_time': datetime.now(),
            'data_types': data_types,
            'steps': {}
        }

        try:
            # Initialize BigQuery dataset and tables
            logger.info("Initializing BigQuery dataset and tables")
            self._initialize_bigquery()

            # Parse data types
            if data_types == 'all':
                data_types_list = ['ohlcv', 'indicators', 'macro', 'temporal', 'metadata']
            else:
                data_types_list = [dt.strip() for dt in data_types.split(',')]

            # Run backfill for each data type
            if 'metadata' in data_types_list:
                logger.info("Step: Populating stock metadata")
                metadata_success = self._backfill_metadata()
                results['steps']['metadata'] = {'success': metadata_success}

            if 'ohlcv' in data_types_list:
                logger.info("Step: Backfilling OHLCV data")
                ohlcv_stats = self._backfill_ohlcv(start_date, end_date, batch_size)
                results['steps']['ohlcv'] = ohlcv_stats

            if 'indicators' in data_types_list:
                logger.info("Step: Calculating technical indicators")
                indicator_stats = self._backfill_indicators(start_date, end_date)
                results['steps']['indicators'] = indicator_stats

            if 'macro' in data_types_list:
                logger.info("Step: Backfilling macro data")
                macro_success = self._backfill_macro()
                results['steps']['macro'] = {'success': macro_success}

            if 'temporal' in data_types_list:
                logger.info("Step: Generating temporal features")
                temporal_success = self._backfill_temporal(start_date, end_date)
                results['steps']['temporal'] = {'success': temporal_success}

            # Create feature view after all data is loaded
            if all(step.get('success', False) for step in results['steps'].values()):
                logger.info("Creating feature view")
                self.bq_client.create_feature_view()

            results['overall_success'] = True

        except Exception as e:
            logger.error(f"Backfill failed: {e}")
            results['overall_success'] = False
            results['error'] = str(e)

        finally:
            results['end_time'] = datetime.now()
            results['duration_hours'] = (
                    (results['end_time'] - results['start_time']).total_seconds() / 3600
            )

            # Log results
            self._log_job_results(results)

            # Print summary
            self._print_summary(results)

        return results

    def _initialize_bigquery(self):
        """Initialize BigQuery dataset and tables."""
        try:
            # Create dataset
            self.bq_client.create_dataset_if_not_exists()

            # Create tables
            self.bq_client.create_tables()

            logger.info("BigQuery initialization complete")

        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            raise

    def _backfill_metadata(self) -> bool:
        """Populate stock metadata table."""
        try:
            metadata_records = []

            for sector, stocks in self.stocks_config.items():
                if sector == 'indices':
                    continue

                for stock in stocks:
                    metadata_records.append({
                        'ticker': stock['ticker'],
                        'name': stock['name'],
                        'sector': sector,
                        'industry': stock.get('sub_industry', ''),
                        'market_cap_category': stock['market_cap'],
                        'exchange': 'NASDAQ',  # Default, could be enhanced
                        'updated_at': datetime.now()
                    })

            import pandas as pd
            metadata_df = pd.DataFrame(metadata_records)

            self.bq_client.insert_dataframe(
                metadata_df,
                'stock_metadata',
                if_exists='replace'
            )

            logger.info(f"Populated metadata for {len(metadata_df)} stocks")
            return True

        except Exception as e:
            logger.error(f"Metadata backfill failed: {e}")
            return False

    def _backfill_ohlcv(
            self,
            start_date: str,
            end_date: str,
            batch_size: int
    ) -> Dict:
        """Backfill OHLCV data with duplicate prevention and incremental support."""
        stats = {
            'success': True,
            'total_tickers': 0,
            'successful_tickers': 0,
            'failed_tickers': [],
            'total_records': 0,
            'skipped_records': 0,
            'new_records': 0
        }

        try:
            # Get all tickers
            all_tickers = []
            for sector, stocks in self.stocks_config.items():
                for stock in stocks:
                    all_tickers.append(stock['ticker'])

            stats['total_tickers'] = len(all_tickers)

            # Process in batches
            for i in range(0, len(all_tickers), batch_size):
                batch_tickers = all_tickers[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch_tickers}")

                # Process each ticker individually for better control
                for ticker in batch_tickers:
                    try:
                        # Get existing data date ranges
                        existing_dates = self._get_existing_dates(
                            ticker, 'raw_ohlcv', start_date, end_date
                        )

                        # Find missing date ranges
                        missing_ranges = self._find_missing_date_ranges(
                            existing_dates, start_date, end_date
                        )

                        if not missing_ranges:
                            logger.info(f"No missing data for {ticker} in date range")
                            stats['successful_tickers'] += 1
                            continue

                        # Fetch only missing data
                        total_ticker_records = 0
                        for range_start, range_end in missing_ranges:
                            logger.info(f"Fetching {ticker} for {range_start} to {range_end}")

                            # Fetch data for this range
                            data = self.ohlcv_fetcher.fetch_historical_data(
                                start_date=range_start,
                                end_date=range_end,
                                tickers=[ticker]
                            )

                            # Store data if fetched successfully
                            if ticker in data and not data[ticker].empty:
                                df = data[ticker]

                                # Additional check to prevent duplicates
                                df = self._remove_duplicate_dates(df, ticker, 'raw_ohlcv')

                                if not df.empty:
                                    self.bq_client.insert_dataframe(
                                        df,
                                        'raw_ohlcv',
                                        if_exists='append'
                                    )
                                    total_ticker_records += len(df)
                                    stats['new_records'] += len(df)
                                    logger.info(f"Stored {len(df)} records for {ticker}")

                        stats['successful_tickers'] += 1
                        stats['total_records'] += total_ticker_records

                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        stats['failed_tickers'].append(ticker)

                # Rate limiting between batches
                time.sleep(60)

            stats['success'] = len(stats['failed_tickers']) < stats['total_tickers'] * 0.1

        except Exception as e:
            logger.error(f"OHLCV backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats

    def _backfill_indicators(
            self,
            start_date: str,
            end_date: str
    ) -> Dict:
        """Backfill technical indicators with duplicate prevention."""
        stats = {
            'success': True,
            'tickers_processed': 0,
            'total_records': 0,
            'new_records': 0
        }

        try:
            # Get unique tickers with OHLCV data
            query = f"""
            SELECT DISTINCT ticker
            FROM `{BQ_TABLES['raw_ohlcv']}`
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            """

            tickers_df = self.bq_client.query(query)

            for ticker in tickers_df['ticker']:
                try:
                    # Check existing indicator data
                    existing_dates = self._get_existing_dates(
                        ticker, 'technical_indicators', start_date, end_date
                    )

                    # Find missing date ranges
                    missing_ranges = self._find_missing_date_ranges(
                        existing_dates, start_date, end_date
                    )

                    if not missing_ranges:
                        logger.info(f"No missing indicators for {ticker}")
                        stats['tickers_processed'] += 1
                        continue

                    # Get OHLCV data (need extra days for indicator calculation)
                    query = f"""
                    SELECT *
                    FROM `{BQ_TABLES['raw_ohlcv']}`
                    WHERE ticker = '{ticker}'
                      AND date >= DATE_SUB('{start_date}', INTERVAL 200 DAY)
                      AND date <= '{end_date}'
                    ORDER BY date
                    """

                    ohlcv_df = self.bq_client.query(query)

                    if len(ohlcv_df) < 20:
                        logger.warning(f"Insufficient data for {ticker}")
                        continue

                    # Calculate indicators
                    indicators_df = self.indicator_calculator.calculate_all_indicators(
                        ohlcv_df,
                        ticker=ticker
                    )

                    # Filter to only missing dates
                    indicators_df = indicators_df[
                        indicators_df['date'].isin(
                            self._get_dates_from_ranges(missing_ranges)
                        )
                    ]

                    # Store to BigQuery
                    if not indicators_df.empty:
                        self.bq_client.insert_dataframe(
                            indicators_df,
                            'technical_indicators',
                            if_exists='append'
                        )
                        stats['tickers_processed'] += 1
                        stats['total_records'] += len(indicators_df)
                        stats['new_records'] += len(indicators_df)
                        logger.info(f"Stored {len(indicators_df)} new indicators for {ticker}")

                except Exception as e:
                    logger.error(f"Failed to process indicators for {ticker}: {e}")

            logger.info(f"Processed indicators for {stats['tickers_processed']} tickers")

        except Exception as e:
            logger.error(f"Indicator backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats

    def _backfill_macro(self) -> bool:
        """Backfill macro economic data."""
        try:
            # Check if we already have recent macro data
            query = f"""
            SELECT MAX(date) as latest_date
            FROM `{BQ_TABLES['macro_data']}`
            """

            result = self.bq_client.query(query)
            if not result.empty and result['latest_date'].iloc[0]:
                latest_date = pd.to_datetime(result['latest_date'].iloc[0])
                if (datetime.now() - latest_date).days < 7:
                    logger.info("Macro data is up to date")
                    return True

            # Fetch all indicators
            indicator_data = self.macro_fetcher.fetch_all_indicators()

            if indicator_data:
                # Aggregate to wide format
                wide_df = self.macro_fetcher.aggregate_to_wide_format(indicator_data)

                # Calculate composite indicators
                wide_df = self.macro_fetcher.calculate_composite_indicators(wide_df)

                # Remove duplicates before storing
                if self.bq_client.table_exists('macro_data'):
                    existing_dates_query = f"""
                    SELECT DISTINCT date
                    FROM `{BQ_TABLES['macro_data']}`
                    """
                    existing_dates = self.bq_client.query(existing_dates_query)['date'].tolist()
                    wide_df = wide_df[~wide_df['date'].isin(existing_dates)]
                    logger.info(f"Filtered out {len(existing_dates)} existing dates")

                # Store to BigQuery
                if not wide_df.empty:
                    return self.macro_fetcher.store_to_bigquery(wide_df)
                else:
                    logger.info("No new macro data to store")
                    return True

            return False

        except Exception as e:
            logger.error(f"Macro backfill failed: {e}")
            return False

    def _backfill_temporal(self, start_date: str, end_date: str) -> bool:
        """Backfill temporal features."""
        try:
            # Check existing temporal features
            if self.bq_client.table_exists('temporal_features'):
                query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM `{BQ_TABLES['temporal_features']}`
                """
                result = self.bq_client.query(query)

                if not result.empty and result['min_date'].iloc[0]:
                    existing_start = pd.to_datetime(result['min_date'].iloc[0])
                    existing_end = pd.to_datetime(result['max_date'].iloc[0])

                    # Check if we need to backfill
                    if (existing_start <= pd.to_datetime(start_date) and
                            existing_end >= pd.to_datetime(end_date)):
                        logger.info("Temporal features already exist for date range")
                        return True

            # Extend date range to include future dates for predictions
            extended_end = (
                    datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=90)
            ).strftime('%Y-%m-%d')

            return self.temporal_engineer.store_temporal_features(
                start_date,
                extended_end
            )

        except Exception as e:
            logger.error(f"Temporal backfill failed: {e}")
            return False

    def _get_existing_dates(
            self,
            ticker: str,
            table_name: str,
            start_date: str,
            end_date: str
    ) -> list:
        """Get list of existing dates for a ticker in a table."""
        try:
            if not self.bq_client.table_exists(table_name):
                return []

            query = f"""
            SELECT DISTINCT date
            FROM `{BQ_TABLES[table_name]}`
            WHERE ticker = '{ticker}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """

            result = self.bq_client.query(query)
            if result.empty:
                return []

            return pd.to_datetime(result['date']).dt.strftime('%Y-%m-%d').tolist()

        except Exception as e:
            logger.warning(f"Error getting existing dates: {e}")
            return []

    def _find_missing_date_ranges(
            self,
            existing_dates: list,
            start_date: str,
            end_date: str
    ) -> list:
        """Find missing date ranges given existing dates."""
        if not existing_dates:
            return [(start_date, end_date)]

        # Convert to datetime for comparison
        existing_dates = pd.to_datetime(existing_dates)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Create full date range
        full_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

        # Find missing dates
        missing_dates = full_range.difference(existing_dates)

        if missing_dates.empty:
            return []

        # Convert to ranges
        ranges = []
        current_start = None

        for i in range(len(missing_dates)):
            if current_start is None:
                current_start = missing_dates[i]

            # Check if next date is consecutive
            if i == len(missing_dates) - 1 or (missing_dates[i+1] - missing_dates[i]).days > 1:
                ranges.append((
                    current_start.strftime('%Y-%m-%d'),
                    missing_dates[i].strftime('%Y-%m-%d')
                ))
                current_start = None

        return ranges

    def _get_dates_from_ranges(self, date_ranges: list) -> list:
        """Convert date ranges to list of dates."""
        all_dates = []

        for start_date, end_date in date_ranges:
            dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq='D'
            )
            all_dates.extend(dates)

        return all_dates

    def _remove_duplicate_dates(
            self,
            df: pd.DataFrame,
            ticker: str,
            table_name: str
    ) -> pd.DataFrame:
        """Remove dates that already exist in BigQuery."""
        try:
            if not self.bq_client.table_exists(table_name):
                return df

            # Get existing dates for this ticker
            dates_str = "', '".join(df['date'].dt.strftime('%Y-%m-%d'))
            query = f"""
            SELECT DISTINCT date
            FROM `{BQ_TABLES[table_name]}`
            WHERE ticker = '{ticker}'
              AND date IN ('{dates_str}')
            """

            existing_df = self.bq_client.query(query)

            if not existing_df.empty:
                existing_dates = pd.to_datetime(existing_df['date'])
                df = df[~df['date'].isin(existing_dates)]
                logger.info(f"Filtered out {len(existing_dates)} duplicate dates for {ticker}")

            return df

        except Exception as e:
            logger.warning(f"Error checking duplicates: {e}")
            return df

    def _print_summary(self, results: Dict):
        """Print backfill summary."""
        logger.info("=" * 60)
        logger.info("BACKFILL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Date range: {results['start_date']} to {results['end_date']}")
        logger.info(f"Duration: {results['duration_hours']:.2f} hours")
        logger.info(f"Overall success: {results['overall_success']}")

        if 'steps' in results:
            logger.info("\nStep Results:")
            for step, stats in results['steps'].items():
                if isinstance(stats, dict):
                    success = stats.get('success', False)
                    logger.info(f"  {step}: {'SUCCESS' if success else 'FAILED'}")

                    if step == 'ohlcv' and 'total_records' in stats:
                        logger.info(f"    - Total records: {stats['total_records']:,}")
                        logger.info(f"    - New records: {stats['new_records']:,}")
                        logger.info(f"    - Tickers: {stats['successful_tickers']}/{stats['total_tickers']}")
                    elif step == 'indicators' and 'total_records' in stats:
                        logger.info(f"    - Total records: {stats['total_records']:,}")
                        logger.info(f"    - New records: {stats['new_records']:,}")
                        logger.info(f"    - Tickers: {stats['tickers_processed']}")

        logger.info("=" * 60)

    def _log_job_results(self, results: Dict):
        """Log job results to BigQuery."""
        try:
            job_log = {
                'job_name': 'backfill',
                'run_date': datetime.now().date(),
                'start_time': results['start_time'],
                'end_time': results['end_time'],
                'duration_hours': results['duration_hours'],
                'success': results['overall_success'],
                'parameters': {
                    'start_date': results['start_date'],
                    'end_date': results['end_date'],
                    'data_types': results['data_types']
                },
                'step_results': results.get('steps', {}),
                'error': results.get('error')
            }

            import pandas as pd
            import json

            log_df = pd.DataFrame([{
                'job_name': job_log['job_name'],
                'run_date': job_log['run_date'],
                'start_time': job_log['start_time'],
                'end_time': job_log['end_time'],
                'duration_hours': job_log['duration_hours'],
                'success': job_log['success'],
                'parameters': json.dumps(job_log['parameters']),
                'step_results': json.dumps(job_log['step_results']),
                'error': job_log['error']
            }])

            self.bq_client.insert_dataframe(
                log_df,
                'job_logs',
                if_exists='append'
            )

        except Exception as e:
            logger.error(f"Failed to log job results: {e}")


@click.command()
@click.option('--start-date', required=True, help='Start date for backfill (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date for backfill (YYYY-MM-DD)')
@click.option('--data-types', default='all', help='Data types to backfill (all,ohlcv,indicators,macro,temporal,metadata)')
@click.option('--batch-size', default=10, help='Batch size for ticker processing')
def main(start_date, end_date, data_types, batch_size):
    """Run historical data backfill job."""
    job = BackfillJob()
    results = job.run(
        start_date=start_date,
        end_date=end_date,
        data_types=data_types,
        batch_size=batch_size
    )

    if results['overall_success']:
        logger.info("Backfill completed successfully")
        sys.exit(0)
    else:
        logger.error("Backfill failed")
        sys.exit(1)


if __name__ == '__main__':
    main()