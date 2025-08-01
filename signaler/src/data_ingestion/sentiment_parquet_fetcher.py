"""
Sentiment data fetcher that writes to Parquet/GCS.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from src.utils import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.storage import ParquetStorageClient
from src.storage.storage_config import (
    GCS_BUCKET,
    get_storage_path,
    get_partition_cols,
    PERFORMANCE_CONFIG
)
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from config.settings import load_stocks_config


class SentimentParquetFetcher:
    """Fetch sentiment data and store in Parquet format on GCS."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET):
        """Initialize sentiment fetcher with storage client."""
        self.storage_client = ParquetStorageClient(bucket_name=bucket_name)
        self.av_client = AlphaVantageClient()
        self.stocks_config = load_stocks_config()
        
        # Performance settings
        self.chunk_size = PERFORMANCE_CONFIG['chunk_size']
        self.max_workers = PERFORMANCE_CONFIG['max_parallel_writes']
        
        logger.info(f"Initialized SentimentParquetFetcher with bucket: {bucket_name}")
    
    def fetch_and_store_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 7
    ) -> Dict[str, any]:
        """
        Fetch sentiment data from Alpha Vantage and store to GCS.
        
        Args:
            symbols: List of symbols to fetch (None = all configured)
            lookback_days: Number of days to look back for sentiment
        
        Returns:
            Summary of ingestion results
        """
        if symbols is None:
            symbols = self._get_all_symbols()
        
        logger.info(f"Starting sentiment fetch for {len(symbols)} symbols")
        
        results = {
            'symbols_processed': 0,
            'symbols_failed': [],
            'total_articles': 0,
            'start_time': datetime.now()
        }
        
        # Process symbols in batches to respect rate limits
        batch_size = 5  # Process 5 symbols at a time
        symbol_batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
        
        all_sentiment_data = []
        
        for batch_idx, batch in enumerate(symbol_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(symbol_batches)}")
            
            for symbol in batch:
                try:
                    # Fetch sentiment data
                    sentiment_df = self.av_client.get_sentiment_data([symbol])
                    
                    if sentiment_df.empty:
                        logger.warning(f"No sentiment data for {symbol}")
                        continue
                    
                    # Process and enrich data
                    sentiment_df = self._process_sentiment_data(sentiment_df, symbol)
                    
                    # Filter by date
                    cutoff_date = datetime.now() - timedelta(days=lookback_days)
                    sentiment_df = sentiment_df[sentiment_df['timestamp'] >= cutoff_date]
                    
                    if not sentiment_df.empty:
                        all_sentiment_data.append(sentiment_df)
                        results['symbols_processed'] += 1
                        results['total_articles'] += len(sentiment_df)
                        logger.info(f"Fetched {len(sentiment_df)} articles for {symbol}")
                    
                except Exception as e:
                    logger.error("Failed to fetch sentiment for {symbol}")
                    results['symbols_failed'].append(symbol)
            
            # Rate limiting between batches
            if batch_idx < len(symbol_batches) - 1:
                logger.info("Rate limiting: waiting 60 seconds...")
                time.sleep(60)
        
        # Combine and store all data
        if all_sentiment_data:
            combined_df = pd.concat(all_sentiment_data, ignore_index=True)
            
            # Store to GCS
            write_result = self.storage_client.write_dataframe(
                df=combined_df,
                path=get_storage_path('sentiment'),
                partition_cols=get_partition_cols('sentiment'),
                compression='snappy'
            )
            
            logger.info(f"Stored {write_result['rows_written']} sentiment records")
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        logger.info(f"Sentiment fetch complete: {results}")
        return results
    
    def update_latest_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        deduplicate: bool = True
    ) -> Dict[str, any]:
        """
        Update with latest sentiment data.
        
        Args:
            symbols: Symbols to update (None = all)
            deduplicate: Whether to check for duplicates
        
        Returns:
            Update results
        """
        if symbols is None:
            symbols = self._get_all_symbols()
        
        logger.info(f"Updating sentiment for {len(symbols)} symbols")
        
        results = {
            'symbols_updated': 0,
            'new_articles': 0,
            'duplicates_skipped': 0,
            'start_time': datetime.now()
        }
        
        # Process each symbol
        for symbol in symbols:
            try:
                # Fetch latest sentiment
                sentiment_df = self.av_client.get_sentiment_data([symbol])
                
                if sentiment_df.empty:
                    continue
                
                # Process data
                sentiment_df = self._process_sentiment_data(sentiment_df, symbol)
                
                # Append with deduplication
                if deduplicate:
                    append_result = self.storage_client.append_dataframe(
                        df=sentiment_df,
                        path=get_storage_path('sentiment'),
                        partition_cols=get_partition_cols('sentiment'),
                        deduplicate_cols=['symbol', 'timestamp', 'title']
                    )
                    
                    results['new_articles'] += append_result.get('rows_appended', 0)
                    results['duplicates_skipped'] += append_result.get('duplicates_skipped', 0)
                else:
                    write_result = self.storage_client.write_dataframe(
                        df=sentiment_df,
                        path=get_storage_path('sentiment'),
                        partition_cols=get_partition_cols('sentiment')
                    )
                    results['new_articles'] += write_result['rows_written']
                
                if sentiment_df.shape[0] > 0:
                    results['symbols_updated'] += 1
                
                # Rate limiting
                time.sleep(12)
                
            except Exception as e:
                logger.error("Failed to update sentiment for {symbol}")
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        logger.info(f"Sentiment update complete: {results}")
        return results
    
    def _process_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process and enrich sentiment DataFrame."""
        if 'time_published' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time_published'])
        elif 'timestamp' not in df.columns:
            df['timestamp'] = pd.Timestamp.now()
        
        # Ensure symbol column
        df['symbol'] = symbol
        
        # Add date partitioning column
        df['date'] = df['timestamp'].dt.date
        
        # Normalize sentiment scores to [-1, 1] range if needed
        score_cols = ['overall_sentiment_score', 'ticker_sentiment_score']
        for col in score_cols:
            if col in df.columns:
                # Alpha Vantage scores are already in [-1, 1] range
                df[col] = df[col].fillna(0)
        
        # Add sentiment categories
        if 'overall_sentiment_score' in df.columns:
            df['sentiment_category'] = pd.cut(
                df['overall_sentiment_score'],
                bins=[-1, -0.35, -0.15, 0.15, 0.35, 1],
                labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
            )
        
        # Add source
        df['source'] = 'alpha_vantage_news'
        
        # Add ingestion metadata
        df['ingested_at'] = datetime.now()
        
        # Select and order columns
        columns = [
            'symbol', 'timestamp', 'date', 'title', 
            'overall_sentiment_score', 'overall_sentiment_label',
            'ticker_sentiment_score', 'ticker_sentiment_label',
            'sentiment_category', 'source', 'ingested_at'
        ]
        
        # Keep only columns that exist
        columns = [col for col in columns if col in df.columns]
        
        return df[columns]
    
    def aggregate_daily_sentiment(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by day for each symbol.
        
        Args:
            symbols: List of symbols (None = all)
            lookback_days: Days to look back
        
        Returns:
            DataFrame with daily aggregated sentiment
        """
        # Read sentiment data
        filters = []
        if symbols:
            # Note: PyArrow doesn't support 'in' operator directly
            # Would need to create multiple OR conditions
            pass
        
        # Add date filter
        start_date = datetime.now().date() - timedelta(days=lookback_days)
        filters.extend([
            ('date', '>=', start_date),
            ('date', '<=', datetime.now().date())
        ])
        
        try:
            df = self.storage_client.read_dataframe(
                path=get_storage_path('sentiment'),
                filters=filters
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Aggregate by symbol and date
            daily_sentiment = df.groupby(['symbol', 'date']).agg({
                'overall_sentiment_score': ['mean', 'std', 'count'],
                'sentiment_category': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = [
                'symbol', 'date', 'avg_sentiment', 'sentiment_std', 
                'article_count', 'dominant_category'
            ]
            
            # Add sentiment strength
            daily_sentiment['sentiment_strength'] = daily_sentiment['avg_sentiment'].abs()
            
            # Add trend (requires at least 3 days of data)
            daily_sentiment['sentiment_trend'] = (
                daily_sentiment.groupby('symbol')['avg_sentiment']
                .rolling(window=3, min_periods=1)
                .apply(lambda x: x[-1] - x[0] if len(x) > 1 else 0)
                .reset_index(level=0, drop=True)
            )
            
            return daily_sentiment
            
        except Exception as e:
            logger.error("Failed to aggregate sentiment")
            return pd.DataFrame()
    
    def get_sentiment_summary(self) -> pd.DataFrame:
        """Get summary of stored sentiment data."""
        try:
            # Get dataset info
            info = self.storage_client.get_dataset_info(
                path=get_storage_path('sentiment')
            )
            
            # Read sample to get date ranges
            df = self.storage_client.read_dataframe(
                path=get_storage_path('sentiment'),
                columns=['symbol', 'timestamp', 'overall_sentiment_score']
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate summary by symbol
            summary = df.groupby('symbol').agg({
                'timestamp': ['min', 'max', 'count'],
                'overall_sentiment_score': ['mean', 'std']
            }).reset_index()
            
            summary.columns = [
                'symbol', 'first_date', 'last_date', 'article_count',
                'avg_sentiment', 'sentiment_volatility'
            ]
            
            summary['days_of_data'] = (
                (summary['last_date'] - summary['first_date']).dt.days
            )
            
            # Add sentiment classification
            summary['sentiment_bias'] = summary['avg_sentiment'].apply(
                lambda x: 'bullish' if x > 0.1 else 'bearish' if x < -0.1 else 'neutral'
            )
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get sentiment summary")
            return pd.DataFrame()
    
    def _get_all_symbols(self) -> List[str]:
        """Get all configured symbols."""
        symbols = []
        for sector, stocks in self.stocks_config.items():
            for stock in stocks:
                symbols.append(stock['ticker'])
        return symbols
    
    def _get_latest_sentiment_date(self, symbol: str) -> Optional[pd.Timestamp]:
        """Get the latest sentiment date for a symbol."""
        try:
            # Read latest partition
            df = self.storage_client.read_latest_partition(
                path=get_storage_path('sentiment'),
                partition_col='date',
                filters=[('symbol', '=', symbol)]
            )
            
            if not df.empty:
                return df['timestamp'].max()
                
        except Exception as e:
            logger.debug(f"No existing sentiment for {symbol}: {e}")
        
        return None