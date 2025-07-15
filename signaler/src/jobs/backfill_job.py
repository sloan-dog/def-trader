"""
Backfill historical data for all data sources.
"""
import click
import sys
from datetime import datetime, timedelta
from loguru import logger
import time

from src.utils.logging_config import setup_logging, log_exception
from src.data_ingestion.ohlcv_fetcher import OHLCVFetcher
from src.data_ingestion.macro_data_fetcher import MacroDataFetcher
from src.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.utils.bigquery import BigQueryClient
from config.settings import load_stocks_config, BQ_TABLES
from typing import Dict, List, Tuple
import pandas as pd


class BackfillJob:
    """Orchestrate historical data backfill."""

    def __init__(self):
        """Initialize backfill components."""
        self.is_cloud_run = self._is_cloud_run()
        self.ohlcv_fetcher = OHLCVFetcher()
        self.macro_fetcher = MacroDataFetcher()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.bq_client = BigQueryClient()
        self.stocks_config = load_stocks_config()

        # Configure logging with JSON format for Google Cloud
        setup_logging(
            level="INFO",
            log_file="logs/backfill_{time}.log" if not self.is_cloud_run else None,
            rotation="100 MB",
            retention="7 days"
        )
    
    def _is_cloud_run(self) -> bool:
        """Check if running in Google Cloud Run."""
        import os
        return bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_PROJECT"))

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
        """Backfill HOURLY OHLCV data with duplicate prevention and incremental support."""
        logger.info(f"Starting HOURLY OHLCV backfill from {start_date} to {end_date}")

        stats = {
            'success': True,
            'total_tickers': 0,
            'successful_tickers': 0,
            'failed_tickers': [],
            'total_records': 0,
            'skipped_records': 0,
            'new_records': 0,
            'api_calls': 0,
            'data_frequency': 'hourly'
        }

        try:
            # Get all tickers
            all_tickers = []
            for sector, stocks in self.stocks_config.items():
                for stock in stocks:
                    all_tickers.append(stock['ticker'])

            stats['total_tickers'] = len(all_tickers)

            # Calculate expected API calls for hourly data
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            months_needed = len(pd.date_range(start=start, end=end, freq='MS'))
            expected_api_calls = len(all_tickers) * months_needed

            logger.info(f"Expected API calls: {expected_api_calls}")
            logger.info(f"Estimated time with 75 RPM: {expected_api_calls / 75:.1f} minutes")

            # Process in batches
            for i in range(0, len(all_tickers), batch_size):
                batch_tickers = all_tickers[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch_tickers}")

                # Process each ticker individually for better control
                for ticker in batch_tickers:
                    try:
                        # Get existing hourly data date ranges
                        existing_hours = self._get_existing_hourly_data(
                            ticker, start_date, end_date
                        )

                        # Find missing months
                        missing_months = self._find_missing_months_hourly(
                            existing_hours, start_date, end_date
                        )

                        if not missing_months:
                            logger.info(f"No missing hourly data for {ticker} in date range")
                            stats['successful_tickers'] += 1
                            continue

                        # Fetch only missing months
                        total_ticker_records = 0
                        for month in missing_months:
                            try:
                                logger.info(f"Fetching {ticker} hourly data for {month}")

                                # Fetch hourly data for this month
                                month_data = self.ohlcv_fetcher.av_client.get_hourly_ohlcv(
                                    ticker,
                                    month=month
                                )

                                stats['api_calls'] += 1

                                # Filter to date range
                                if not month_data.empty:
                                    month_data = month_data[
                                        (month_data['date'] >= pd.to_datetime(start_date).date()) &
                                        (month_data['date'] <= pd.to_datetime(end_date).date())
                                        ]

                                    # Additional check to prevent duplicates
                                    month_data = self._remove_duplicate_hourly_data(
                                        month_data, ticker
                                    )

                                    if not month_data.empty:
                                        # Store to BigQuery
                                        self.bq_client.insert_dataframe(
                                            month_data,
                                            'raw_ohlcv_hourly',
                                            if_exists='append'
                                        )
                                        total_ticker_records += len(month_data)
                                        stats['new_records'] += len(month_data)
                                        logger.info(f"Stored {len(month_data)} hourly records for {ticker} in {month}")

                            except Exception as e:
                                logger.error(f"Failed to fetch {ticker} for {month}: {e}")

                        stats['successful_tickers'] += 1
                        stats['total_records'] += total_ticker_records
                        logger.info(f"Completed {ticker}: {total_ticker_records} total hourly records")

                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        stats['failed_tickers'].append(ticker)

            stats['success'] = len(stats['failed_tickers']) < stats['total_tickers'] * 0.1

            logger.info(f"HOURLY backfill completed: {stats['successful_tickers']}/{stats['total_tickers']} tickers")
            logger.info(f"Total API calls made: {stats['api_calls']}")

        except Exception as e:
            logger.error(f"HOURLY OHLCV backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats

    def _backfill_indicators(
            self,
            start_date: str,
            end_date: str
    ) -> Dict:
        """Backfill technical indicators for HOURLY data."""
        stats = {
            'success': True,
            'tickers_processed': 0,
            'total_records': 0,
            'new_records': 0,
            'data_frequency': 'hourly'
        }

        try:
            # Get unique tickers with hourly OHLCV data
            query = f"""
            SELECT DISTINCT ticker
            FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            """

            tickers_df = self.bq_client.query(query)

            for ticker in tickers_df['ticker']:
                try:
                    # Check existing hourly indicator data
                    existing_hours = self._get_existing_hourly_indicators(
                        ticker, start_date, end_date
                    )

                    # Find missing hours
                    missing_hours = self._find_missing_hours_for_indicators(
                        existing_hours, ticker, start_date, end_date
                    )

                    if not missing_hours:
                        logger.info(f"No missing hourly indicators for {ticker}")
                        stats['tickers_processed'] += 1
                        continue

                    # Get hourly OHLCV data (need extra days for indicator calculation)
                    query = f"""
                    SELECT *
                    FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
                    WHERE ticker = '{ticker}'
                      AND datetime >= DATETIME_SUB('{start_date} 00:00:00', INTERVAL 30 DAY)
                      AND datetime <= DATETIME_ADD('{end_date} 23:59:59', INTERVAL 1 DAY)
                    ORDER BY datetime
                    """

                    ohlcv_df = self.bq_client.query(query)

                    if len(ohlcv_df) < 200:  # Need enough hourly data for indicators
                        logger.warning(f"Insufficient hourly data for {ticker} indicators")
                        continue

                    # Calculate hourly indicators
                    indicators_df = self.indicator_calculator.calculate_hourly_indicators(
                        ohlcv_df,
                        ticker=ticker
                    )

                    # Filter to only missing hours
                    indicators_df = indicators_df[
                        indicators_df['datetime'].isin(missing_hours)
                    ]

                    # Store to BigQuery
                    if not indicators_df.empty:
                        self.bq_client.insert_dataframe(
                            indicators_df,
                            'technical_indicators_hourly',
                            if_exists='append'
                        )
                        stats['tickers_processed'] += 1
                        stats['total_records'] += len(indicators_df)
                        stats['new_records'] += len(indicators_df)
                        logger.info(f"Stored {len(indicators_df)} new hourly indicators for {ticker}")

                except Exception as e:
                    logger.error(f"Failed to process hourly indicators for {ticker}: {e}")

            logger.info(f"Processed hourly indicators for {stats['tickers_processed']} tickers")

        except Exception as e:
            logger.error(f"Hourly indicator backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats


    def _get_existing_hourly_indicators(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get existing hourly indicator data."""
        query = f"""
        SELECT datetime
        FROM `{BQ_TABLES['technical_indicators_hourly']}`
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY datetime
        """

        return self.bq_client.query(query)


    def _find_missing_hours_for_indicators(
            self,
            existing_hours: pd.DataFrame,
            ticker: str,
            start_date: str,
            end_date: str
    ) -> List[datetime]:
        """Find missing hours that need indicators calculated."""
        # Get all hours that have OHLCV data
        query = f"""
        SELECT DISTINCT datetime
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY datetime
        """

        ohlcv_hours = self.bq_client.query(query)

        if ohlcv_hours.empty:
            return []

        all_hours = pd.to_datetime(ohlcv_hours['datetime'])

        if existing_hours.empty:
            return all_hours.tolist()

        existing_hours_set = set(pd.to_datetime(existing_hours['datetime']))
        missing_hours = [h for h in all_hours if h not in existing_hours_set]

        return missing_hours
        """
        Calculate technical indicators for hourly data.
        
        Note: Period parameters need adjustment for hourly vs daily:
        - Daily SMA 20 = ~20 days = ~140-260 hourly periods
        - Daily SMA 50 = ~50 days = ~350-650 hourly periods
        - Daily RSI 14 = ~14 days = ~98-182 hourly periods
        """

        # Ensure data is sorted by datetime
        df = df.sort_values('datetime').copy()

        # Adjusted periods for hourly data (assuming 7 trading hours per day)
        HOURS_PER_DAY = 7

        # Simple Moving Averages - adjusted for hourly
        df['sma_20h'] = df['close'].rolling(window=20).mean()  # 20-hour SMA
        df['sma_140h'] = df['close'].rolling(window=140).mean()  # ~20-day equivalent
        df['sma_350h'] = df['close'].rolling(window=350).mean()  # ~50-day equivalent

        # Exponential Moving Averages
        df['ema_12h'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_84h'] = df['close'].ewm(span=84, adjust=False).mean()  # ~12-day equivalent
        df['ema_182h'] = df['close'].ewm(span=182, adjust=False).mean()  # ~26-day equivalent

        # RSI - using 14-hour period and ~14-day equivalent
        df['rsi_14h'] = self._calculate_rsi(df['close'], period=14)
        df['rsi_98h'] = self._calculate_rsi(df['close'], period=98)

        # MACD - adjusted for hourly
        exp1 = df['close'].ewm(span=84, adjust=False).mean()   # ~12-day
        exp2 = df['close'].ewm(span=182, adjust=False).mean()  # ~26-day
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=63, adjust=False).mean()  # ~9-day
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands - using 20-hour and ~20-day periods
        df['bb_middle_20h'] = df['close'].rolling(window=20).mean()
        df['bb_upper_20h'] = df['bb_middle_20h'] + 2 * df['close'].rolling(window=20).std()
        df['bb_lower_20h'] = df['bb_middle_20h'] - 2 * df['close'].rolling(window=20).std()

        # Volume-based indicators
        df['volume_sma_20h'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20h']

        # Intraday-specific indicators
        df['hour_of_day'] = pd.to_datetime(df['datetime']).dt.hour
        df['is_market_open'] = df['hour_of_day'].between(9, 15)  # Regular trading hours
        df['is_first_hour'] = df['hour_of_day'] == 9
        df['is_last_hour'] = df['hour_of_day'] == 15

        # High/Low of day (calculated per trading day)
        df['high_of_day'] = df.groupby('date')['high'].transform('max')
        df['low_of_day'] = df.groupby('date')['low'].transform('min')
        df['pct_from_high'] = (df['close'] - df['high_of_day']) / df['high_of_day'] * 100
        df['pct_from_low'] = (df['close'] - df['low_of_day']) / df['low_of_day'] * 100

        # VWAP (Volume Weighted Average Price) - reset daily
        df['vwap'] = (df.groupby('date')
                      .apply(lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum())
                      .reset_index(level=0, drop=True))

        # Add ticker
        df['ticker'] = ticker

        return df


    def get_indicator_config_hourly(self):
        """Get indicator configuration for hourly data."""
        return {
            'moving_averages': {
                'sma': [20, 140, 350],  # 20-hour, ~20-day, ~50-day
                'ema': [12, 84, 182]    # 12-hour, ~12-day, ~26-day
            },
            'oscillators': {
                'rsi_periods': [14, 98],  # 14-hour, ~14-day
                'stochastic_period': 14
            },
            'volatility': {
                'atr_period': 14,
                'bollinger_period': 20,
                'bollinger_std': 2
            },
            'volume': {
                'volume_sma': [20, 140]
            }
        }

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

    def _backfill_hourly_ohlcv(
            self,
            start_date: str,
            end_date: str,
            batch_size: int = 5  # Smaller batches due to more API calls needed
    ) -> Dict:
        """
        Backfill hourly OHLCV data.

        Note: This is much slower than daily data due to API limitations:
        - Need to fetch month-by-month
        - Each ticker-month combination is one API call
        - Rate limited to 5 calls per minute
        """
        logger.info(f"Starting hourly OHLCV backfill from {start_date} to {end_date}")

        stats = {
            'success': True,
            'total_tickers': 0,
            'successful_tickers': 0,
            'failed_tickers': [],
            'total_records': 0,
            'total_api_calls': 0,
            'estimated_time_hours': 0
        }

        try:
            # Get all tickers
            all_tickers = self._get_all_tickers()
            stats['total_tickers'] = len(all_tickers)

            # Calculate number of months and estimate time
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            months_per_ticker = len(pd.date_range(start=start, end=end, freq='MS'))
            total_api_calls = len(all_tickers) * months_per_ticker
            stats['total_api_calls'] = total_api_calls

            # At 5 calls/minute with 12 second delays
            estimated_minutes = (total_api_calls * 12) / 60
            stats['estimated_time_hours'] = round(estimated_minutes / 60, 2)

            logger.info(f"Estimated time: {stats['estimated_time_hours']} hours "
                        f"({total_api_calls} API calls at 5/minute)")

            # Process each ticker
            for idx, ticker in enumerate(all_tickers):
                try:
                    logger.info(f"Processing {ticker} ({idx+1}/{len(all_tickers)})")

                    # Check existing data to avoid redundant fetches
                    existing_ranges = self._get_existing_hourly_ranges(ticker, start_date, end_date)
                    missing_months = self._find_missing_months(existing_ranges, start_date, end_date)

                    if not missing_months:
                        logger.info(f"No missing hourly data for {ticker}")
                        stats['successful_tickers'] += 1
                        continue

                    # Fetch missing months
                    ticker_data = []
                    for year_month in missing_months:
                        try:
                            month_data = self.ohlcv_fetcher.av_client.get_hourly_ohlcv(
                                ticker,
                                month=year_month
                            )

                            # Filter to date range
                            month_data = month_data[
                                (month_data['date'] >= pd.to_datetime(start_date).date()) &
                                (month_data['date'] <= pd.to_datetime(end_date).date())
                                ]

                            if not month_data.empty:
                                ticker_data.append(month_data)
                                logger.info(f"Fetched {len(month_data)} hourly records for {ticker} in {year_month}")

                            # Rate limiting
                            time.sleep(12)

                        except Exception as e:
                            logger.error(f"Failed to fetch {ticker} for {year_month}: {e}")

                    # Combine and store all data for ticker
                    if ticker_data:
                        combined_df = pd.concat(ticker_data, ignore_index=True)

                        # Store to BigQuery
                        success = self.ohlcv_fetcher.store_hourly_to_bigquery({ticker: combined_df})

                        if success.get(ticker):
                            stats['successful_tickers'] += 1
                            stats['total_records'] += len(combined_df)
                        else:
                            stats['failed_tickers'].append(ticker)
                    else:
                        stats['failed_tickers'].append(ticker)

                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {e}")
                    stats['failed_tickers'].append(ticker)

            logger.info(f"Hourly backfill completed: {stats['successful_tickers']}/{stats['total_tickers']} tickers")

        except Exception as e:
            logger.error(f"Hourly OHLCV backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats


    def _get_existing_hourly_ranges(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get existing hourly data date ranges for a ticker."""
        query = f"""
        SELECT 
            DATE(datetime) as date,
            COUNT(*) as hourly_records
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY date
        ORDER BY date
        """
        return self.bq_client.query(query)


    def _find_missing_months(
            self,
            existing_ranges: pd.DataFrame,
            start_date: str,
            end_date: str
    ) -> List[str]:
        """Find months that need to be fetched."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate all months in range
        all_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y-%m').tolist()

        if existing_ranges.empty:
            return all_months

        # Find months with incomplete data (less than expected hourly records)
        # Assuming ~7-13 records per trading day, ~150-250 per month
        existing_months = set()
        for _, row in existing_ranges.iterrows():
            if row['hourly_records'] >= 100:  # Threshold for "complete" month
                month = pd.to_datetime(row['date']).strftime('%Y-%m')
                existing_months.add(month)

        missing_months = [m for m in all_months if m not in existing_months]

        return missing_months

    def _backfill_ohlcv(
            self,
            start_date: str,
            end_date: str,
            batch_size: int
    ) -> Dict:
        """Backfill HOURLY OHLCV data with duplicate prevention and incremental support."""
        logger.info(f"Starting HOURLY OHLCV backfill from {start_date} to {end_date}")

        stats = {
            'success': True,
            'total_tickers': 0,
            'successful_tickers': 0,
            'failed_tickers': [],
            'total_records': 0,
            'skipped_records': 0,
            'new_records': 0,
            'api_calls': 0,
            'data_frequency': 'hourly'
        }

        try:
            # Get all tickers
            all_tickers = []
            for sector, stocks in self.stocks_config.items():
                for stock in stocks:
                    all_tickers.append(stock['ticker'])

            stats['total_tickers'] = len(all_tickers)

            # Calculate expected API calls for hourly data
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            months_needed = len(pd.date_range(start=start, end=end, freq='MS'))
            expected_api_calls = len(all_tickers) * months_needed

            logger.info(f"Expected API calls: {expected_api_calls}")
            logger.info(f"Estimated time with 75 RPM: {expected_api_calls / 75:.1f} minutes")

            # Process in batches
            for i in range(0, len(all_tickers), batch_size):
                batch_tickers = all_tickers[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch_tickers}")

                # Process each ticker individually for better control
                for ticker in batch_tickers:
                    try:
                        # Get existing hourly data date ranges
                        existing_hours = self._get_existing_hourly_data(
                            ticker, start_date, end_date
                        )

                        # Find missing months
                        missing_months = self._find_missing_months_hourly(
                            existing_hours, start_date, end_date
                        )

                        if not missing_months:
                            logger.info(f"No missing hourly data for {ticker} in date range")
                            stats['successful_tickers'] += 1
                            continue

                        # Fetch only missing months
                        total_ticker_records = 0
                        for month in missing_months:
                            try:
                                logger.info(f"Fetching {ticker} hourly data for {month}")

                                # Fetch hourly data for this month
                                month_data = self.ohlcv_fetcher.av_client.get_hourly_ohlcv(
                                    ticker,
                                    month=month
                                )

                                stats['api_calls'] += 1

                                # Filter to date range
                                if not month_data.empty:
                                    month_data = month_data[
                                        (month_data['date'] >= pd.to_datetime(start_date).date()) &
                                        (month_data['date'] <= pd.to_datetime(end_date).date())
                                        ]

                                    # Additional check to prevent duplicates
                                    month_data = self._remove_duplicate_hourly_data(
                                        month_data, ticker
                                    )

                                    if not month_data.empty:
                                        # Store to BigQuery
                                        self.bq_client.insert_dataframe(
                                            month_data,
                                            'raw_ohlcv_hourly',
                                            if_exists='append'
                                        )
                                        total_ticker_records += len(month_data)
                                        stats['new_records'] += len(month_data)
                                        logger.info(f"Stored {len(month_data)} hourly records for {ticker} in {month}")

                            except Exception as e:
                                logger.error(f"Failed to fetch {ticker} for {month}: {e}")

                        stats['successful_tickers'] += 1
                        stats['total_records'] += total_ticker_records
                        logger.info(f"Completed {ticker}: {total_ticker_records} total hourly records")

                    except Exception as e:
                        logger.error(f"Failed to process {ticker}: {e}")
                        stats['failed_tickers'].append(ticker)

            stats['success'] = len(stats['failed_tickers']) < stats['total_tickers'] * 0.1

            logger.info(f"HOURLY backfill completed: {stats['successful_tickers']}/{stats['total_tickers']} tickers")
            logger.info(f"Total API calls made: {stats['api_calls']}")

        except Exception as e:
            logger.error(f"HOURLY OHLCV backfill failed: {e}")
            stats['success'] = False
            stats['error'] = str(e)

        return stats


    def _get_existing_hourly_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get existing hourly data for a ticker."""
        query = f"""
        SELECT 
            DATE(datetime) as date,
            EXTRACT(HOUR FROM datetime) as hour,
            COUNT(*) as record_count
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE ticker = '{ticker}'
          AND date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY date, hour
        ORDER BY date, hour
        """

        result = self.bq_client.query(query)
        return result if not result.empty else pd.DataFrame()

    def _find_missing_months_hourly(
            self,
            existing_data: pd.DataFrame,
            start_date: str,
            end_date: str
    ) -> List[str]:
        """Find months that need hourly data to be fetched."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generate all months in range
        all_months = pd.date_range(start=start, end=end, freq='MS').strftime('%Y-%m').tolist()

        # Include end month if not already in list
        end_month = end.strftime('%Y-%m')
        if end_month not in all_months:
            all_months.append(end_month)

        if existing_data.empty:
            return all_months

        # Find months with incomplete hourly data
        # Expected: ~7-13 hours per trading day, ~150-280 records per month
        MIN_RECORDS_PER_MONTH = 100

        existing_months = set()
        for month_str in all_months:
            month_data = existing_data[
                existing_data['date'].astype(str).str.startswith(month_str)
            ]

            if len(month_data) >= MIN_RECORDS_PER_MONTH:
                existing_months.add(month_str)

        missing_months = [m for m in all_months if m not in existing_months]

        return missing_months


    def _remove_duplicate_hourly_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove duplicate hourly records that already exist in BigQuery."""
        if df.empty:
            return df

        min_datetime = df['datetime'].min()
        max_datetime = df['datetime'].max()

        # Query existing datetimes
        query = f"""
        SELECT DISTINCT datetime
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE ticker = '{ticker}'
          AND datetime BETWEEN '{min_datetime}' AND '{max_datetime}'
        """

        existing_df = self.bq_client.query(query)

        if not existing_df.empty:
            existing_datetimes = pd.to_datetime(existing_df['datetime'])
            initial_count = len(df)
            df = df[~df['datetime'].isin(existing_datetimes)]
            removed_count = initial_count - len(df)

            if removed_count > 0:
                logger.info(f"Filtered out {removed_count} duplicate hourly records for {ticker}")

        return df


    def _get_existing_dates(self, ticker: str, table: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get existing dates for a ticker in a table."""
        # For hourly data, we need to check by datetime, not just date
        if table == 'technical_indicators_hourly':
            query = f"""
            SELECT DISTINCT datetime
            FROM `{BQ_TABLES[table]}`
            WHERE ticker = '{ticker}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY datetime
            """
        else:
            query = f"""
            SELECT DISTINCT date
            FROM `{BQ_TABLES[table]}`
            WHERE ticker = '{ticker}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
            """

        result = self.bq_client.query(query)
        return result if not result.empty else pd.DataFrame()


    def _find_missing_date_ranges(
            self,
            existing_dates: pd.DataFrame,
            start_date: str,
            end_date: str
    ) -> List[Tuple[str, str]]:
        """Find continuous ranges of missing dates."""
        if existing_dates.empty:
            return [(start_date, end_date)]

        # For hourly data, we need different logic
        if 'datetime' in existing_dates.columns:
            # Group by date and check if we have enough hours
            dates_with_hours = existing_dates.groupby(pd.to_datetime(existing_dates['datetime']).dt.date).size()
            dates_with_sufficient_hours = dates_with_hours[dates_with_hours >= 7].index  # At least 7 hours
            existing_date_list = pd.to_datetime(dates_with_sufficient_hours)
        else:
            existing_date_list = pd.to_datetime(existing_dates['date'])

        # Generate all business days in range
        all_dates = pd.date_range(start_date, end_date, freq='B')

        # Find missing dates
        missing_dates = all_dates.difference(existing_date_list)

        if missing_dates.empty:
            return []

        # Group consecutive missing dates into ranges
        missing_ranges = []
        current_start = missing_dates[0]
        current_end = missing_dates[0]

        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - current_end).days <= 1:
                current_end = missing_dates[i]
            else:
                missing_ranges.append((
                    current_start.strftime('%Y-%m-%d'),
                    current_end.strftime('%Y-%m-%d')
                ))
                current_start = missing_dates[i]
                current_end = missing_dates[i]

        # Add the last range
        missing_ranges.append((
            current_start.strftime('%Y-%m-%d'),
            current_end.strftime('%Y-%m-%d')
        ))

        return missing_ranges




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