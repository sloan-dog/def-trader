"""
Backfill job for Cloud Run Jobs.
Supports hourly, daily, weekly, and historical backfills.
"""

import click
import sys
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger

from src.shared_logging import setup_logging, log_exception
from src.jobs.historical_backfill_job import HistoricalBackfillJob
from src.utils.bigquery.client import BigQueryClient
from src.utils.bigquery.backfill_tracker import BackfillTracker


class BackfillJob:
    """Cloud Run Job for backfill operations."""
    
    def __init__(self):
        """Initialize job components."""
        self.is_cloud_run = self._is_cloud_run()
        
        # Configure logging with JSON format for Google Cloud
        setup_logging(
            level="INFO",
            log_file="logs/backfill_{time}.log" if not self.is_cloud_run else None,
            rotation="1 day",
            retention="30 days",
            app_name="signaler-backfill"
        )
        
        self.bq_client = BigQueryClient()
        self.tracker = BackfillTracker(self.bq_client)
    
    def _is_cloud_run(self) -> bool:
        """Check if running in Google Cloud Run."""
        import os
        return bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_PROJECT"))

    def run_hourly_backfill(self):
        """Run hourly backfill - last 2 days of OHLCV data."""
        try:
            logger.info("Starting hourly backfill")
            
            # Calculate date range (last 2 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=2)
            
            # Run backfill for OHLCV data only
            self._run_backfill(
                start_date=start_date,
                end_date=end_date,
                data_types=["ohlcv"],
                batch_size=20,
                backfill_type="hourly"
            )
            
            logger.info("Hourly backfill completed successfully")
            return True
            
        except Exception as e:
            log_exception("Hourly backfill failed", exception=e)
            return False

    def run_daily_backfill(self):
        """Run daily backfill - last 7 days of data."""
        try:
            logger.info("Starting daily backfill")
            
            # Calculate date range (last 7 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            
            # Run backfill for OHLCV and macro data
            self._run_backfill(
                start_date=start_date,
                end_date=end_date,
                data_types=["ohlcv", "macro"],
                batch_size=15,
                backfill_type="daily"
            )
            
            logger.info("Daily backfill completed successfully")
            return True
            
        except Exception as e:
            log_exception("Daily backfill failed", exception=e)
            return False

    def run_weekly_backfill(self):
        """Run weekly backfill - last 30 days of data."""
        try:
            logger.info("Starting weekly backfill")
            
            # Calculate date range (last 30 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            # Run backfill for all data types
            self._run_backfill(
                start_date=start_date,
                end_date=end_date,
                data_types=["ohlcv", "macro"],
                batch_size=10,
                backfill_type="weekly"
            )
            
            logger.info("Weekly backfill completed successfully")
            return True
            
        except Exception as e:
            log_exception("Weekly backfill failed", exception=e)
            return False

    def run_historical_backfill(
        self,
        start_year: int,
        end_year: int,
        data_types: List[str] = ["ohlcv"],
        batch_size: int = 10,
        backfill_id: Optional[str] = None
    ):
        """Run historical backfill for specified years."""
        try:
            logger.info(f"Starting historical backfill: {start_year}-{end_year}")
            
            # Create historical backfill job
            job = HistoricalBackfillJob(
                start_year=start_year,
                end_year=end_year,
                data_types=data_types,
                batch_size=batch_size,
                backfill_id=backfill_id
            )
            
            # Run the historical backfill
            job.run()
            
            logger.info("Historical backfill completed successfully")
            return True
            
        except Exception as e:
            log_exception("Historical backfill failed", exception=e)
            return False

    def _run_backfill(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        data_types: List[str],
        batch_size: int,
        backfill_type: str
    ):
        """Run a backfill operation."""
        logger.info(f"Running {backfill_type} backfill: {start_date} to {end_date}")
        logger.info(f"Data types: {data_types}, Batch size: {batch_size}")
        
        # Import here to avoid circular imports
        from src.jobs.backfill_job import BackfillJob as LegacyBackfillJob
        
        # Create legacy backfill job (reuse existing logic)
        job = LegacyBackfillJob()
        
        # Run the backfill
        results = job.run_backfill(
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            batch_size=batch_size
        )
        
        logger.info(f"{backfill_type.capitalize()} backfill results: {results}")
        return results


@click.command()
@click.option('--type', 'backfill_type', 
              type=click.Choice(['hourly', 'daily', 'weekly', 'historical']),
              required=True,
              help='Type of backfill to run')
@click.option('--start-year', type=int, help='Start year for historical backfill')
@click.option('--end-year', type=int, help='End year for historical backfill')
@click.option('--data-types', multiple=True, 
              default=['ohlcv'],
              help='Data types to backfill (can specify multiple)')
@click.option('--batch-size', type=int, default=10, help='Batch size for processing')
@click.option('--backfill-id', help='Backfill ID for historical backfill')
def main(backfill_type, start_year, end_year, data_types, batch_size, backfill_id):
    """Run backfill job."""
    try:
        job = BackfillJob()
        
        if backfill_type == 'hourly':
            success = job.run_hourly_backfill()
        elif backfill_type == 'daily':
            success = job.run_daily_backfill()
        elif backfill_type == 'weekly':
            success = job.run_weekly_backfill()
        elif backfill_type == 'historical':
            if not start_year or not end_year:
                logger.error("Start year and end year are required for historical backfill")
                sys.exit(1)
            success = job.run_historical_backfill(
                start_year=start_year,
                end_year=end_year,
                data_types=list(data_types),
                batch_size=batch_size,
                backfill_id=backfill_id
            )
        else:
            logger.error(f"Unknown backfill type: {backfill_type}")
            sys.exit(1)
        
        if success:
            logger.info(f"{backfill_type.capitalize()} backfill completed successfully")
            sys.exit(0)
        else:
            logger.error(f"{backfill_type.capitalize()} backfill failed")
            sys.exit(1)
            
    except Exception as e:
        log_exception(f"Fatal error in {backfill_type} backfill job", exception=e)
        sys.exit(1)


if __name__ == '__main__':
    main()