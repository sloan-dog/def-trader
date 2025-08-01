"""
OHLCV data fetcher that writes to Parquet/GCS.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.utils import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.relativedelta import relativedelta

from src.storage import ParquetStorageClient
from src.storage.storage_config import (
    GCS_BUCKET,
    get_storage_path,
    get_partition_cols,
    PERFORMANCE_CONFIG
)
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from config.settings import load_stocks_config, INGESTION_CONFIG


class OHLCVParquetFetcher:
    """Fetch OHLCV data and store in Parquet format on GCS."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET):
        """Initialize fetcher with storage client."""
        self.storage_client = ParquetStorageClient(bucket_name=bucket_name)
        self.av_client = AlphaVantageClient()
        self.stocks_config = load_stocks_config()
        
        # Performance settings
        self.chunk_size = PERFORMANCE_CONFIG['chunk_size']
        self.max_workers = PERFORMANCE_CONFIG['max_parallel_writes']
        
        logger.info(f"Initialized OHLCVParquetFetcher with bucket: {bucket_name}")
    
    def fetch_and_store_historical(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '60min'
    ) -> Dict[str, any]:
        """
        Fetch historical OHLCV data and store to GCS.
        
        Args:
            symbols: List of symbols to fetch (None = all configured)
            start_date: Start date for historical data
            end_date: End date for historical data  
            interval: Data interval (1min, 5min, 15min, 30min, 60min, daily)
        
        Returns:
            Summary of ingestion results
        """
        # Set defaults
        if symbols is None:
            symbols = self._get_all_symbols()
        if start_date is None:
            start_date = INGESTION_CONFIG.get('backfill_start_date', '2023-01-01')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting historical fetch for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {
            'symbols_processed': 0,
            'symbols_failed': [],
            'total_rows': 0,
            'start_time': datetime.now()
        }
        
        # Process symbols with rate limiting
        for symbol in symbols:
            retries = 0
            max_retries = INGESTION_CONFIG.get('max_retries', 3)
            retry_delay = INGESTION_CONFIG.get('retry_delay', 5)
            
            while retries <= max_retries:
                try:
                    logger.info(f"Fetching {symbol}..." + (f" (retry {retries}/{max_retries})" if retries > 0 else ""))
                    
                    # Fetch from Alpha Vantage
                    # For historical backfill with date range, use month-by-month fetching
                    if start_date and interval == '60min':
                        df = self._fetch_historical_months(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date or datetime.now().strftime('%Y-%m-%d'),
                            interval=interval
                        )
                    else:
                        # For other intervals or no date range, use regular fetch
                        df = self._fetch_symbol_data(
                            symbol=symbol,
                            interval=interval,
                            outputsize='full'
                        )
                    
                    if df is None or df.empty:
                        logger.warning(f"No data returned for {symbol}")
                        results['symbols_failed'].append(symbol)
                        break

                    logger.info(f"Fetched dataframe for {symbol}, {df.head()}")
                    
                    # Filter date range
                    df = self._filter_date_range(df, start_date, end_date)
                    
                    if df.empty:
                        logger.warning(f"No data in date range for {symbol}")
                        continue
                    
                    # Prepare for storage
                    df = self._prepare_dataframe(df, symbol, interval)

                    logger.info(f"Prepared dataframe for {symbol}, {df.head()}")
                    
                    # Store to GCS
                    write_result = self.storage_client.write_dataframe(
                        df=df,
                        path=f"{get_storage_path('ohlcv')}/{interval}",
                        partition_cols=get_partition_cols('ohlcv'),
                        compression='snappy'
                    )
                    
                    results['symbols_processed'] += 1
                    results['total_rows'] += write_result['rows_written']
                    
                    logger.info(f"Stored {write_result['rows_written']} rows for {symbol}")
                    
                    # Rate limiting handled by AlphaVantageClient (75 calls/min for premium)
                    break  # Success, exit retry loop
                
                except ValueError as e:
                    if "API call frequency limit reached" in str(e):
                        retries += 1
                        if retries <= max_retries:
                            logger.warning(f"Rate limit hit for {symbol}, waiting {retry_delay}s before retry {retries}/{max_retries}")
                            time.sleep(retry_delay)
                            continue
                    logger.error(f"Failed to process {symbol}: {e}")
                    results['symbols_failed'].append(symbol)
                    break
                    
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    results['symbols_failed'].append(symbol)
                    break
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        logger.info(f"Historical fetch complete: {results}")
        return results
    
    def update_latest_data(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = 24,
        interval: str = '60min'
    ) -> Dict[str, any]:
        """
        Update with latest OHLCV data (for daily/hourly updates).
        
        Args:
            symbols: Symbols to update (None = all)
            lookback_hours: Hours to look back for updates
            interval: Data interval
        
        Returns:
            Update results
        """
        if symbols is None:
            symbols = self._get_all_symbols()
        
        logger.info(f"Updating latest data for {len(symbols)} symbols")
        
        results = {
            'symbols_updated': 0,
            'new_rows': 0,
            'duplicates_skipped': 0,
            'start_time': datetime.now()
        }
        
        # Process each symbol
        for symbol in symbols:
            retries = 0
            max_retries = INGESTION_CONFIG.get('max_retries', 3)
            retry_delay = INGESTION_CONFIG.get('retry_delay', 5)
            
            while retries <= max_retries:
                try:
                    # Get latest stored timestamp
                    latest_timestamp = self._get_latest_timestamp(symbol, interval)
                    
                    # Fetch recent data
                    df = self._fetch_symbol_data(
                        symbol=symbol,
                        interval=interval,
                        outputsize='compact'  # Last 100 data points
                    )
                    
                    if df is None or df.empty:
                        break
                    
                    # Filter to new data only
                    if latest_timestamp:
                        df = df[df['timestamp'] > latest_timestamp]
                    
                    if df.empty:
                        logger.debug(f"No new data for {symbol}")
                        continue
                    
                    # Prepare and append
                    df = self._prepare_dataframe(df, symbol, interval)
                    
                    append_result = self.storage_client.append_dataframe(
                        df=df,
                        path=f"{get_storage_path('ohlcv')}/{interval}",
                        partition_cols=get_partition_cols('ohlcv'),
                        deduplicate_cols=['symbol', 'timestamp']
                    )
                    
                    results['symbols_updated'] += 1
                    results['new_rows'] += append_result.get('rows_appended', 0)
                    results['duplicates_skipped'] += append_result.get('duplicates_skipped', 0)
                    
                    # Rate limiting handled by AlphaVantageClient
                    break  # Success, exit retry loop
                    
                except ValueError as e:
                    if "API call frequency limit reached" in str(e):
                        retries += 1
                        if retries <= max_retries:
                            logger.warning(f"Rate limit hit for {symbol}, waiting {retry_delay}s before retry {retries}/{max_retries}")
                            time.sleep(retry_delay)
                            continue
                    logger.error(f"Failed to update {symbol}: {e}")
                    break
                    
                except Exception as e:
                    logger.error(f"Failed to update {symbol}: {e}")
                    break
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        logger.info(f"Update complete: {results}")
        return results
    
    def batch_update_parallel(
        self,
        symbols: List[str],
        interval: str = '60min',
        max_workers: int = None
    ) -> Dict[str, any]:
        """
        Update multiple symbols in parallel (respecting rate limits).
        
        Args:
            symbols: List of symbols
            interval: Data interval
            max_workers: Number of parallel workers
        
        Returns:
            Batch update results
        """
        if max_workers is None:
            max_workers = min(self.max_workers, 5)  # Rate limit constraint
        
        logger.info(f"Starting parallel update for {len(symbols)} symbols")
        
        results = {
            'total_symbols': len(symbols),
            'success': 0,
            'failed': [],
            'total_rows': 0
        }
        
        # Split symbols into batches based on rate limit
        batch_size = 5  # 5 calls per minute
        symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        for batch_idx, batch in enumerate(symbol_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(symbol_batches)}")
            
            batch_data = {}
            
            # Fetch data for batch
            for symbol in batch:
                try:
                    df = self._fetch_symbol_data(symbol, interval, 'compact')
                    if df is not None and not df.empty:
                        df = self._prepare_dataframe(df, symbol, interval)
                        batch_data[symbol] = df
                        results['success'] += 1
                    else:
                        results['failed'].append(symbol)
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}")
                    results['failed'].append(symbol)
            
            # Write batch in parallel
            if batch_data:
                write_results = self.storage_client.parallel_write(
                    dataframes=batch_data,
                    base_path=f"{get_storage_path('ohlcv')}/{interval}",
                    partition_cols=get_partition_cols('ohlcv'),
                    max_workers=max_workers
                )
                
                for symbol, write_result in write_results.items():
                    if 'error' not in write_result:
                        results['total_rows'] += write_result.get('rows_written', 0)
            
            # Wait before next batch (rate limiting)
            if batch_idx < len(symbol_batches) - 1:
                logger.info("Waiting 60s for rate limit...")
                time.sleep(60)
        
        return results
    
    def _fetch_symbol_data(
        self,
        symbol: str,
        interval: str,
        outputsize: str,
        month: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        try:
            # For hourly data (60min), always use intraday endpoint
            # Only use daily endpoint if explicitly requesting daily bars
            if interval == 'daily':
                df = self.av_client.get_daily_ohlcv(symbol, outputsize)
            elif interval == '60min' and month:
                # Use the hourly endpoint with month parameter for historical data
                df = self.av_client.get_hourly_ohlcv(symbol, month=month)
            else:
                # This handles 1min, 5min, 15min, 30min, 60min (recent data)
                df = self.av_client.get_intraday_data(symbol, interval, outputsize)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}" + (f" for month {month}" if month else ""))
            return None
    
    def _fetch_historical_months(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '60min'
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data month by month for a symbol."""
        if interval != '60min':
            # For non-hourly intervals, use the regular fetch
            return self._fetch_symbol_data(symbol, interval, 'full')
        
        # Parse dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Limit to Alpha Vantage's historical data availability (January 2000)
        av_start = pd.to_datetime('2000-01-01')
        if start_dt < av_start:
            logger.info(f"Adjusting start date from {start_date} to 2000-01-01 (Alpha Vantage limit)")
            start_dt = av_start
        
        all_data = []
        current_month = start_dt
        
        # Generate list of months to fetch
        while current_month <= end_dt:
            month_str = current_month.strftime('%Y-%m')
            logger.info(f"Fetching {symbol} for {month_str}...")
            
            # Fetch this month's data
            df = self._fetch_symbol_data(symbol, interval, 'full', month=month_str)
            
            if df is not None and not df.empty:
                all_data.append(df)
                logger.debug(f"Got {len(df)} rows for {symbol} in {month_str}")
            else:
                logger.warning(f"No data for {symbol} in {month_str}")
            
            # Move to next month
            current_month = current_month + relativedelta(months=1)
            
            # Rate limiting is handled by AlphaVantageClient
        
        if not all_data:
            return None
        
        # Combine all monthly data
        # First, ensure all DataFrames have their index reset to avoid issues
        reset_data = []
        for df in all_data:
            if isinstance(df.index, pd.DatetimeIndex):
                df_reset = df.reset_index(drop=True)
            else:
                df_reset = df.copy()
            reset_data.append(df_reset)
        
        combined_df = pd.concat(reset_data, ignore_index=True)
        
        # Remove duplicates based on datetime column (which Alpha Vantage provides)
        if 'datetime' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['datetime'])
            combined_df = combined_df.sort_values('datetime')
        elif 'timestamp' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp')
        
        logger.info(f"Combined {len(all_data)} months of data for {symbol}: {len(combined_df)} total rows")
        return combined_df
    
    def _prepare_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> pd.DataFrame:
        """Prepare DataFrame for storage with proper schema."""
        # Debug: log the columns we received
        logger.debug(f"DataFrame columns for {symbol}: {df.columns.tolist()}")
        logger.debug(f"DataFrame index name: {df.index.name}")
        
        # Ensure we have timestamp column
        if 'timestamp' not in df.columns and df.index.name in ['date', 'timestamp', 'datetime']:
            df = df.reset_index()
        
        # Drop the existing 'date' column if it exists (Alpha Vantage provides it)
        if 'date' in df.columns and 'datetime' in df.columns:
            df = df.drop(columns=['date'])
        
        # Rename columns to standard names
        column_mapping = {
            'datetime': 'timestamp',  # For intraday data from Alpha Vantage
            '1. open': 'open',
            '2. high': 'high', 
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend',
            '8. split coefficient': 'split_coefficient'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add symbol if not present
        if 'symbol' not in df.columns:
            df['symbol'] = symbol
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.error(f"No timestamp column found for {symbol}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError("Missing timestamp column")
        
        # Add partition columns
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        
        # Add metadata
        df['interval'] = interval
        df['ingested_at'] = datetime.now()
        
        # Select and order columns
        columns = [
            'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'adjusted_close', 'dividend', 'split_coefficient',
            'date', 'year', 'month', 'day', 'interval', 'ingested_at'
        ]
        
        # Keep only columns that exist
        columns = [col for col in columns if col in df.columns]
        
        return df[columns]
    
    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Filter DataFrame to date range."""
        # Check which date column exists
        date_col = 'timestamp' if 'timestamp' in df.columns else 'datetime'
        df[date_col] = pd.to_datetime(df[date_col])
        mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
        return df[mask].copy()
    
    def _get_latest_timestamp(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.Timestamp]:
        """Get the latest timestamp for a symbol."""
        try:
            # Read latest partition
            df = self.storage_client.read_latest_partition(
                path=f"{get_storage_path('ohlcv')}/{interval}",
                partition_col='day',
                filters=[('symbol', '=', symbol)]
            )
            
            if not df.empty:
                return df['timestamp'].max()
                
        except Exception as e:
            logger.debug(f"No existing data for {symbol}: {e}")
        
        return None
    
    def _get_all_symbols(self) -> List[str]:
        """Get all configured symbols."""
        symbols = []
        for sector, stocks in self.stocks_config.items():
            for stock in stocks:
                symbols.append(stock['ticker'])
        return symbols
    
    def get_data_summary(self, interval: str = '60min') -> pd.DataFrame:
        """Get summary of stored OHLCV data."""
        try:
            # Get dataset info
            info = self.storage_client.get_dataset_info(
                path=f"{get_storage_path('ohlcv')}/{interval}"
            )
            
            # Read sample to get date ranges
            df = self.storage_client.read_dataframe(
                path=f"{get_storage_path('ohlcv')}/{interval}",
                columns=['symbol', 'timestamp']
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate summary
            summary = df.groupby('symbol').agg({
                'timestamp': ['min', 'max', 'count']
            }).reset_index()
            
            summary.columns = ['symbol', 'first_date', 'last_date', 'row_count']
            summary['days_of_data'] = (
                (summary['last_date'] - summary['first_date']).dt.days
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get data summary")
            return pd.DataFrame()