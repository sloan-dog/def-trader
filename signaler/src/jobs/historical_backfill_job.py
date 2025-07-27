"""
Historical backfill job with progress tracking and resumability.
Designed to backfill years of historical data efficiently.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Optional
import pandas as pd
from loguru import logger

from src.shared_logging import setup_logging
from src.utils.bigquery.client import BigQueryClient
from src.utils.bigquery.backfill_tracker import BackfillTracker
from src.data_ingestion.ohlcv_fetcher import OHLCVFetcher
from src.data_ingestion.macro_data_fetcher import MacroDataFetcher

setup_logging()


class HistoricalBackfillJob:
    """Manages historical data backfill with progress tracking."""
    
    def __init__(
        self,
        start_year: int,
        end_year: int,
        data_types: List[str] = ["ohlcv"],
        batch_size: int = 10,
        backfill_id: Optional[str] = None
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.data_types = data_types
        self.batch_size = batch_size
        
        # Generate backfill ID if not provided
        self.backfill_id = backfill_id or f"historical_{start_year}_{end_year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize clients
        self.bq_client = BigQueryClient()
        self.tracker = BackfillTracker(self.bq_client)
        
        # Initialize fetchers
        if "ohlcv" in data_types:
            self.ohlcv_fetcher = OHLCVFetcher()
        if "macro" in data_types:
            self.macro_fetcher = MacroDataFetcher()
        
        logger.info(f"Initialized historical backfill job: {self.backfill_id}")
    
    def run(self, resume: bool = True):
        """Run the historical backfill job."""
        try:
            # Start or resume backfill
            if resume:
                status = self.tracker.get_backfill_status(self.backfill_id)
                if status and status['status'] == 'completed':
                    logger.info(f"Backfill {self.backfill_id} already completed")
                    return
                elif status and status['status'] == 'in_progress':
                    logger.info(f"Resuming backfill {self.backfill_id}")
                else:
                    self._start_new_backfill()
            else:
                self._start_new_backfill()
            
            # Process months backward in time
            self._process_historical_data()
            
        except Exception as e:
            logger.error(f"Historical backfill failed: {str(e)}")
            self.tracker.fail_backfill(self.backfill_id, str(e))
            raise
    
    def _start_new_backfill(self):
        """Initialize a new backfill job."""
        self.tracker.start_backfill(
            self.backfill_id,
            self.start_year,
            self.end_year,
            self.data_types
        )
        logger.info(f"Started new backfill: {self.start_year}-{self.end_year}")
    
    def _process_historical_data(self):
        """Process historical data month by month, moving backward."""
        while True:
            # Get next month to process
            next_month = self.tracker.get_next_month_to_process(self.backfill_id)
            if not next_month:
                logger.info("No more months to process - backfill complete!")
                break
            
            year, month = next_month
            logger.info(f"Processing {year}-{month:02d}")
            
            # Process each data type
            for data_type in self.data_types:
                if data_type == "ohlcv":
                    self._process_ohlcv_month(year, month)
                elif data_type == "macro":
                    self._process_macro_month(year, month)
            
            # Update progress
            self.tracker.update_progress(self.backfill_id, year, month)
            
            # Log progress
            status = self.tracker.get_backfill_status(self.backfill_id)
            progress_pct = (status['completed_months'] / status['total_months']) * 100
            logger.info(
                f"Progress: {status['completed_months']}/{status['total_months']} "
                f"({progress_pct:.1f}%) - Current: {year}-{month:02d}"
            )
    
    def _process_ohlcv_month(self, year: int, month: int):
        """Process OHLCV data for a specific month."""
        # Get tickers
        tickers_df = self.bq_client.query("""
            SELECT DISTINCT symbol 
            FROM sp500_constituents 
            WHERE symbol NOT LIKE '%.%'
            ORDER BY symbol
        """)
        tickers = tickers_df['symbol'].tolist()
        
        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}: {batch}")
            
            for symbol in batch:
                try:
                    # Check if already completed
                    completed = self.tracker.get_completed_months(
                        self.backfill_id, symbol, 'ohlcv'
                    )
                    if (year, month) in completed:
                        logger.debug(f"Skipping {symbol} {year}-{month:02d} (already completed)")
                        continue
                    
                    # Fetch hourly data
                    month_str = f"{year}-{month:02d}"
                    hourly_data = self.ohlcv_fetcher.av_client.get_hourly_ohlcv(
                        symbol, month_str
                    )
                    
                    if hourly_data is not None and not hourly_data.empty:
                        # Store to BigQuery - wrap in dictionary as expected by store method
                        data_dict = {symbol: hourly_data}
                        self.ohlcv_fetcher.store_hourly_to_bigquery(data_dict)
                        records = len(hourly_data)
                        logger.info(f"Stored {records} hourly records for {symbol} {month_str}")
                        
                        # Checkpoint success
                        self.tracker.checkpoint_month(
                            self.backfill_id, symbol, year, month,
                            'ohlcv', records
                        )
                    else:
                        logger.warning(f"No data for {symbol} {month_str}")
                        # Still checkpoint as completed with 0 records
                        self.tracker.checkpoint_month(
                            self.backfill_id, symbol, year, month,
                            'ohlcv', 0
                        )
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol} {year}-{month:02d}: {str(e)}")
                    # Checkpoint failure
                    self.tracker.checkpoint_month(
                        self.backfill_id, symbol, year, month,
                        'ohlcv', 0, str(e)
                    )
    
    def _process_macro_month(self, year: int, month: int):
        """Process macro data for a specific month."""
        # Macro data is typically fetched differently (not per symbol)
        # This is a placeholder - implement based on your macro data needs
        logger.info(f"Processing macro data for {year}-{month:02d}")
        # TODO: Implement macro data backfill
        pass


def main():
    """CLI entry point for historical backfill."""
    parser = argparse.ArgumentParser(
        description="Run historical data backfill with progress tracking"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="Start year for backfill (e.g., 1995)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help="End year for backfill (e.g., 2024)"
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["ohlcv"],
        choices=["ohlcv", "macro"],
        help="Types of data to backfill"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of tickers to process in parallel"
    )
    parser.add_argument(
        "--backfill-id",
        type=str,
        help="Unique ID for this backfill (for resuming)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if previous progress exists"
    )
    
    args = parser.parse_args()
    
    # Validate years
    if args.start_year > args.end_year:
        logger.error("Start year must be before end year")
        sys.exit(1)
    
    if args.start_year < 1990:
        logger.error("Start year must be 1990 or later")
        sys.exit(1)
    
    current_year = datetime.now().year
    if args.end_year > current_year:
        logger.error(f"End year cannot be after {current_year}")
        sys.exit(1)
    
    # Run the job
    job = HistoricalBackfillJob(
        start_year=args.start_year,
        end_year=args.end_year,
        data_types=args.data_types,
        batch_size=args.batch_size,
        backfill_id=args.backfill_id
    )
    
    job.run(resume=not args.no_resume)
    logger.info("Historical backfill completed!")


if __name__ == "__main__":
    main()