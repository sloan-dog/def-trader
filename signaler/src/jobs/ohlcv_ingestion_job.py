"""
OHLCV data ingestion job for Parquet/GCS storage.
"""
import click
import sys
import time
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional, List

from src.data_ingestion.ohlcv_parquet_fetcher import OHLCVParquetFetcher
from src.storage.storage_config import GCS_BUCKET
from config.settings import load_stocks_config


class OHLCVIngestionJob:
    """Orchestrate OHLCV data ingestion to Parquet/GCS."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET):
        """Initialize the ingestion job."""
        self.fetcher = OHLCVParquetFetcher(bucket_name=bucket_name)
        self.stocks_config = load_stocks_config()
        
        # Set up logging
        logger.add(
            "logs/ohlcv_ingestion_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
    
    def run_historical_backfill(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '60min'
    ) -> bool:
        """
        Run historical backfill for specified symbols.
        
        Args:
            symbols: List of symbols (None = all)
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            interval: Time interval
        
        Returns:
            Success boolean
        """
        logger.info(f"Starting historical backfill - interval: {interval}")
        
        try:
            results = self.fetcher.fetch_and_store_historical(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            # Log summary
            logger.info(f"""
            Historical Backfill Complete:
            - Symbols processed: {results['symbols_processed']}
            - Symbols failed: {len(results['symbols_failed'])}
            - Total rows: {results['total_rows']:,}
            - Duration: {results['duration']:.1f} seconds
            """)
            
            if results['symbols_failed']:
                logger.warning(f"Failed symbols: {results['symbols_failed']}")
            
            return results['symbols_processed'] > 0
            
        except Exception as e:
            logger.error(f"Historical backfill failed: {e}")
            return False
    
    def run_daily_update(
        self,
        symbols: Optional[List[str]] = None,
        lookback_hours: int = 24,
        interval: str = '60min'
    ) -> bool:
        """
        Run daily update for latest data.
        
        Args:
            symbols: List of symbols (None = all)
            lookback_hours: Hours to look back
            interval: Time interval
        
        Returns:
            Success boolean
        """
        logger.info(f"Starting daily update - lookback: {lookback_hours}h")
        
        try:
            results = self.fetcher.update_latest_data(
                symbols=symbols,
                lookback_hours=lookback_hours,
                interval=interval
            )
            
            logger.info(f"""
            Daily Update Complete:
            - Symbols updated: {results['symbols_updated']}
            - New rows: {results['new_rows']:,}
            - Duplicates skipped: {results['duplicates_skipped']:,}
            - Duration: {results['duration']:.1f} seconds
            """)
            
            return results['symbols_updated'] > 0
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return False
    
    def run_batch_update(
        self,
        sector: Optional[str] = None,
        interval: str = '60min'
    ) -> bool:
        """
        Run batch update for a sector or all symbols.
        
        Args:
            sector: Sector name (None = all sectors)
            interval: Time interval
        
        Returns:
            Success boolean
        """
        # Get symbols
        if sector:
            if sector not in self.stocks_config:
                logger.error(f"Unknown sector: {sector}")
                return False
            symbols = [stock['ticker'] for stock in self.stocks_config[sector]]
            logger.info(f"Batch update for sector {sector}: {len(symbols)} symbols")
        else:
            symbols = []
            for stocks in self.stocks_config.values():
                symbols.extend([stock['ticker'] for stock in stocks])
            logger.info(f"Batch update for all sectors: {len(symbols)} symbols")
        
        try:
            results = self.fetcher.batch_update_parallel(
                symbols=symbols,
                interval=interval
            )
            
            logger.info(f"""
            Batch Update Complete:
            - Success: {results['success']}/{results['total_symbols']}
            - Failed: {len(results['failed'])}
            - Total rows: {results['total_rows']:,}
            """)
            
            if results['failed']:
                logger.warning(f"Failed symbols: {results['failed']}")
            
            return results['success'] > 0
            
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            return False
    
    def show_data_summary(self, interval: str = '60min'):
        """Display summary of stored data."""
        logger.info("Fetching data summary...")
        
        try:
            summary = self.fetcher.get_data_summary(interval)
            
            if summary.empty:
                logger.info("No data found")
                return
            
            # Display summary
            logger.info(f"\n{summary.to_string()}")
            
            # Overall stats
            total_rows = summary['row_count'].sum()
            total_symbols = len(summary)
            earliest_date = summary['first_date'].min()
            latest_date = summary['last_date'].max()
            
            logger.info(f"""
            Overall Statistics:
            - Total symbols: {total_symbols}
            - Total rows: {total_rows:,}
            - Date range: {earliest_date} to {latest_date}
            """)
            
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")


@click.group()
def cli():
    """OHLCV data ingestion tool for Parquet/GCS storage."""
    pass


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to fetch (can specify multiple)')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--interval', default='60min', help='Time interval (1min, 5min, 15min, 30min, 60min, daily)')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def backfill(symbols, start_date, end_date, interval, bucket):
    """Run historical backfill for OHLCV data."""
    job = OHLCVIngestionJob(bucket_name=bucket)
    
    # Convert symbols tuple to list
    symbols_list = list(symbols) if symbols else None
    
    success = job.run_historical_backfill(
        symbols=symbols_list,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to update')
@click.option('--lookback', default=24, help='Hours to look back')
@click.option('--interval', default='60min', help='Time interval')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def update(symbols, lookback, interval, bucket):
    """Update with latest OHLCV data."""
    job = OHLCVIngestionJob(bucket_name=bucket)
    
    symbols_list = list(symbols) if symbols else None
    
    success = job.run_daily_update(
        symbols=symbols_list,
        lookback_hours=lookback,
        interval=interval
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--sector', help='Sector to update (omit for all)')
@click.option('--interval', default='60min', help='Time interval')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def batch(sector, interval, bucket):
    """Run batch update for a sector or all symbols."""
    job = OHLCVIngestionJob(bucket_name=bucket)
    
    success = job.run_batch_update(
        sector=sector,
        interval=interval
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--interval', default='60min', help='Time interval')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def summary(interval, bucket):
    """Show summary of stored OHLCV data."""
    job = OHLCVIngestionJob(bucket_name=bucket)
    job.show_data_summary(interval)


@cli.command()
@click.option('--interval', default='60min', help='Time interval')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def continuous(interval, bucket):
    """Run continuous updates (every hour)."""
    job = OHLCVIngestionJob(bucket_name=bucket)
    
    logger.info("Starting continuous OHLCV updates (hourly)")
    
    while True:
        try:
            # Run update
            logger.info(f"Running update at {datetime.now()}")
            job.run_daily_update(lookback_hours=2, interval=interval)
            
            # Wait for next hour
            next_run = datetime.now() + timedelta(hours=1)
            next_run = next_run.replace(minute=5, second=0)  # Run at :05
            
            wait_seconds = (next_run - datetime.now()).total_seconds()
            logger.info(f"Next update at {next_run} ({wait_seconds:.0f}s)")
            
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("Continuous updates stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous update: {e}")
            logger.info("Waiting 5 minutes before retry...")
            time.sleep(300)


if __name__ == '__main__':
    cli()