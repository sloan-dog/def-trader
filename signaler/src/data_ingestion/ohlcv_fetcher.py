"""
OHLCV data fetcher for historical and daily updates.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.shared_logging import setup_logging, log_exception
from config.settings import (
    load_stocks_config,
    INGESTION_CONFIG,
    BQ_TABLES
)
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from src.utils.bigquery import BigQueryClient
from src.data_ingestion.data_validator import DataValidator


class OHLCVFetcher:
    """Fetch OHLCV data for stocks."""

    def __init__(self):
        """Initialize fetcher."""
        self.av_client = AlphaVantageClient()
        self.bq_client = BigQueryClient()
        self.validator = DataValidator()
        self.stocks_config = load_stocks_config()

    def fetch_historical_data(
            self,
            start_date: str = None,
            end_date: str = None,
            tickers: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for multiple tickers.

        Args:
            start_date: Start date for historical data
            end_date: End date for historical data
            tickers: List of tickers to fetch (if None, fetch all configured)

        Returns:
            Dictionary of DataFrames by ticker
        """
        start_date = start_date or INGESTION_CONFIG['backfill_start_date']
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        # Get tickers to fetch
        if tickers is None:
            tickers = self._get_all_tickers()

        logger.info(f"Fetching historical data for {len(tickers)} tickers "
                    f"from {start_date} to {end_date}")

        results = {}
        failed_tickers = []

        # Fetch data with rate limiting
        with ThreadPoolExecutor(max_workers=1) as executor:  # Single thread due to API limits
            future_to_ticker = {
                executor.submit(self._fetch_ticker_data, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        # Filter by date range
                        data['date'] = pd.to_datetime(data['date'])
                        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
                        filtered_data = data[mask].copy()

                        if not filtered_data.empty:
                            results[ticker] = filtered_data
                            logger.info(f"Fetched {len(filtered_data)} records for {ticker}")
                    else:
                        failed_tickers.append(ticker)

                except Exception as e:
                    log_exception(f"Failed to fetch {ticker}", exception=e)
                    failed_tickers.append(ticker)

        if failed_tickers:
            logger.warning(f"Failed to fetch data for: {failed_tickers}")

        return results

    def fetch_daily_updates(
            self,
            tickers: List[str] = None,
            lookback_days: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch recent OHLCV data for daily updates.

        Args:
            tickers: List of tickers to update
            lookback_days: Number of days to look back

        Returns:
            Dictionary of DataFrames by ticker
        """
        if tickers is None:
            tickers = self._get_all_tickers()

        logger.info(f"Starting daily updates for {len(tickers)} tickers")
        results = {}

        for ticker in tickers:
            try:
                logger.debug(f"Processing daily update for {ticker}")
                
                # Get latest date in BigQuery
                latest_date = self.bq_client.get_latest_date('raw_ohlcv', ticker)

                if latest_date:
                    # Fetch data since last update
                    start_date = (latest_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    logger.debug(f"Latest date for {ticker}: {latest_date}, fetching from {start_date}")
                else:
                    # No data exists, fetch last N days
                    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                    logger.debug(f"No existing data for {ticker}, fetching last {lookback_days} days from {start_date}")

                # Fetch recent data
                data = self._fetch_ticker_data(ticker, outputsize='compact')

                if data is not None and not data.empty:
                    # Filter to get only new data
                    data['date'] = pd.to_datetime(data['date'])
                    mask = data['date'] > pd.to_datetime(start_date)
                    new_data = data[mask].copy()

                    if not new_data.empty:
                        results[ticker] = new_data
                        logger.info(f"Fetched {len(new_data)} new records for {ticker}",
                                    ticker=ticker,
                                    records_fetched=len(new_data),
                                    date_range=f"{new_data['date'].min()} to {new_data['date'].max()}")
                    else:
                        logger.info(f"No new data found for {ticker} after {start_date}",
                                    ticker=ticker,
                                    start_date=start_date)
                else:
                    logger.warning(f"No data returned for {ticker} during daily update", ticker=ticker)

            except Exception as e:
                log_exception(f"Failed to fetch daily update for {ticker}", exception=e, ticker=ticker)

        logger.info(f"Completed daily updates. Successfully fetched data for {len(results)} out of {len(tickers)} tickers",
                    successful_tickers=len(results),
                    total_tickers=len(tickers),
                    success_rate=len(results)/len(tickers) if tickers else 0)
        return results

    def store_to_bigquery(
            self,
            data_dict: Dict[str, pd.DataFrame],
            validate: bool = True
    ) -> Dict[str, bool]:
        """
        Store OHLCV data to BigQuery with duplicate prevention.

        Args:
            data_dict: Dictionary of DataFrames by ticker
            validate: Whether to validate data before storing

        Returns:
            Dictionary of success status by ticker
        """
        results = {}

        for ticker, df in data_dict.items():
            try:
                # Validate data
                if validate:
                    is_valid, issues = self.validator.validate_ohlcv(df)
                    if not is_valid:
                        logger.warning(f"Validation issues for {ticker}: {issues}",
                                      ticker=ticker,
                                      validation_issues=issues,
                                      record_count=len(df))
                        # Continue with storage but log the issues

                # Ensure required columns
                df = self._prepare_for_storage(df)

                # Check for existing data to prevent duplicates
                if self.bq_client.table_exists('raw_ohlcv') and not df.empty:
                    # Get date range from dataframe
                    min_date = df['date'].min()
                    max_date = df['date'].max()

                    # Query existing dates
                    query = f"""
                    SELECT DISTINCT date
                    FROM `{BQ_TABLES['raw_ohlcv']}`
                    WHERE ticker = '{ticker}'
                      AND date BETWEEN '{min_date}' AND '{max_date}'
                    """

                    existing_dates_df = self.bq_client.query(query)

                    if not existing_dates_df.empty:
                        existing_dates = pd.to_datetime(existing_dates_df['date']).dt.date
                        initial_count = len(df)
                        df = df[~df['date'].isin(existing_dates)]
                        removed_count = initial_count - len(df)

                        if removed_count > 0:
                            logger.info(f"Filtered out {removed_count} duplicate dates for {ticker}",
                                        ticker=ticker,
                                        duplicates_removed=removed_count,
                                        initial_count=initial_count,
                                        final_count=len(df))

                        if df.empty:
                            logger.info(f"All data for {ticker} already exists, skipping",
                                        ticker=ticker,
                                        existing_dates_count=len(existing_dates))
                            results[ticker] = True
                            continue

                # Store to BigQuery
                if not df.empty:
                    self.bq_client.insert_dataframe(
                        df,
                        'raw_ohlcv',
                        chunk_size=INGESTION_CONFIG['chunk_size'],
                        if_exists='append'
                    )

                    results[ticker] = True
                    logger.info(f"Stored {len(df)} records for {ticker}",
                                ticker=ticker,
                                records_stored=len(df),
                                date_range=f"{df['date'].min()} to {df['date'].max()}")
                else:
                    results[ticker] = True
                    logger.info(f"No new data to store for {ticker}", ticker=ticker)

            except Exception as e:
                log_exception(f"Failed to store data for {ticker}", exception=e, ticker=ticker)
                results[ticker] = False

        return results

    def _fetch_ticker_data(
            self,
            ticker: str,
            outputsize: str = 'full'
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker."""
        logger.info(f"Fetching data for ticker: {ticker} (outputsize: {outputsize})")
        
        try:
            data = self.av_client.get_daily_ohlcv(ticker, outputsize)
            
            if data is not None and not data.empty:
                logger.info(f"Successfully fetched {len(data)} records for {ticker}")
                logger.debug(f"Date range for {ticker}: {data['date'].min()} to {data['date'].max()}")
            else:
                logger.warning(f"Empty or None data returned for {ticker}")
                
            return data
        except Exception as e:
            log_exception(f"Error fetching {ticker}", exception=e, ticker=ticker)
            return None

    def _get_all_tickers(self) -> List[str]:
        """Get all configured tickers."""
        tickers = []

        for sector, stocks in self.stocks_config.items():
            for stock_info in stocks:
                tickers.append(stock_info['ticker'])

        return tickers

    def _prepare_for_storage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for BigQuery storage."""
        # Ensure date column is properly formatted
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove any duplicates
        df = df.drop_duplicates(subset=['ticker', 'date'], keep='last')

        # Sort by date
        df = df.sort_values('date')

        return df

    def backfill_missing_data(
            self,
            check_window_days: int = 30
    ) -> Dict[str, int]:
        """
        Check for and backfill missing data.

        Args:
            check_window_days: Number of days to check for gaps

        Returns:
            Dictionary of number of missing days filled by ticker
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=check_window_days)

        results = {}

        for ticker in self._get_all_tickers():
            try:
                # Query existing data
                query = f"""
                SELECT date
                FROM `{BQ_TABLES['raw_ohlcv']}`
                WHERE ticker = '{ticker}'
                  AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                """

                existing_df = self.bq_client.query(query)

                if existing_df.empty:
                    # No data exists, fetch all
                    data = self._fetch_ticker_data(ticker, outputsize='compact')
                    if data is not None and not data.empty:
                        self.store_to_bigquery({ticker: data})
                        results[ticker] = len(data)
                else:
                    # Check for gaps
                    existing_dates = pd.to_datetime(existing_df['date']).dt.date
                    date_range = pd.date_range(start_date, end_date, freq='B')  # Business days
                    expected_dates = date_range.date

                    missing_dates = set(expected_dates) - set(existing_dates)

                    if missing_dates:
                        logger.info(f"Found {len(missing_dates)} missing dates for {ticker}")

                        # Fetch recent data
                        data = self._fetch_ticker_data(ticker, outputsize='compact')

                        if data is not None and not data.empty:
                            # Filter to missing dates only
                            data['date'] = pd.to_datetime(data['date']).dt.date
                            missing_data = data[data['date'].isin(missing_dates)]

                            if not missing_data.empty:
                                self.store_to_bigquery({ticker: missing_data})
                                results[ticker] = len(missing_data)

            except Exception as e:
                log_exception(f"Failed to backfill {ticker}", exception=e, ticker=ticker)
                results[ticker] = 0

        return results

    def get_data_quality_report(self) -> pd.DataFrame:
        """Generate data quality report for OHLCV data."""
        reports = []

        for ticker in self._get_all_tickers():
            try:
                query = f"""
                SELECT 
                    '{ticker}' as ticker,
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(DISTINCT date) as unique_dates,
                    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                    SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume,
                    SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume,
                    AVG(high - low) / AVG(close) * 100 as avg_daily_range_pct,
                    STDDEV(close) / AVG(close) * 100 as price_volatility_pct
                FROM `{BQ_TABLES['raw_ohlcv']}`
                WHERE ticker = '{ticker}'
                """

                result = self.bq_client.query(query)
                if not result.empty:
                    reports.append(result.iloc[0].to_dict())

            except Exception as e:
                log_exception(f"Failed to get quality report for {ticker}", exception=e, ticker=ticker)

        if reports:
            return pd.DataFrame(reports)
        return pd.DataFrame()


class OHLCVUpdater:
    """Orchestrator for OHLCV data updates."""

    def __init__(self):
        """Initialize updater."""
        self.fetcher = OHLCVFetcher()

    def run_historical_backfill(
            self,
            start_date: str = None,
            end_date: str = None,
            batch_size: int = 10
    ):
        """Run historical data backfill."""
        logger.info("Starting historical data backfill")

        # Get all tickers
        tickers = self.fetcher._get_all_tickers()

        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch_tickers}")

            # Fetch data
            data = self.fetcher.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                tickers=batch_tickers
            )

            # Store to BigQuery
            if data:
                results = self.fetcher.store_to_bigquery(data)
                success_count = sum(1 for success in results.values() if success)
                logger.info(f"Stored data for {success_count}/{len(batch_tickers)} tickers")

            # Rate limiting pause between batches
            time.sleep(60)  # 1 minute between batches

        logger.info("Historical backfill completed")

    def run_daily_update(self):
        """Run daily data update."""
        logger.info("Starting daily OHLCV update")

        # Fetch recent updates
        data = self.fetcher.fetch_daily_updates()

        if data:
            # Store to BigQuery
            results = self.fetcher.store_to_bigquery(data)
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Updated data for {success_count}/{len(data)} tickers")
        else:
            logger.info("No new data to update")

        # Check for and fill any gaps
        missing_filled = self.fetcher.backfill_missing_data(check_window_days=7)
        if missing_filled:
            logger.info(f"Backfilled missing data: {missing_filled}")

        # Generate quality report
        quality_report = self.fetcher.get_data_quality_report()
        if not quality_report.empty:
            # Store quality report
            self.fetcher.bq_client.insert_dataframe(
                quality_report,
                'data_quality_reports',
                if_exists='append'
            )

        logger.info("Daily update completed")