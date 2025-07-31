"""
Sentiment data ingestion job for Parquet/GCS storage.
"""
import click
import sys
import time
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional, List

from src.data_ingestion.sentiment_parquet_fetcher import SentimentParquetFetcher
from src.storage.storage_config import GCS_BUCKET
from config.settings import load_stocks_config


class SentimentIngestionJob:
    """Orchestrate sentiment data ingestion to Parquet/GCS."""
    
    def __init__(self, bucket_name: str = GCS_BUCKET):
        """Initialize the sentiment ingestion job."""
        self.fetcher = SentimentParquetFetcher(bucket_name=bucket_name)
        self.stocks_config = load_stocks_config()
        
        # Set up logging
        logger.add(
            "logs/sentiment_ingestion_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
    
    def run_sentiment_fetch(
        self,
        symbols: Optional[List[str]] = None,
        lookback_days: int = 7
    ) -> bool:
        """
        Fetch sentiment data for symbols.
        
        Args:
            symbols: List of symbols (None = all)
            lookback_days: Days to look back
        
        Returns:
            Success boolean
        """
        logger.info(f"Starting sentiment fetch - lookback: {lookback_days} days")
        
        try:
            results = self.fetcher.fetch_and_store_sentiment(
                symbols=symbols,
                lookback_days=lookback_days
            )
            
            logger.info(f"""
            Sentiment Fetch Complete:
            - Symbols processed: {results['symbols_processed']}
            - Symbols failed: {len(results['symbols_failed'])}
            - Total articles: {results['total_articles']:,}
            - Duration: {results['duration']:.1f} seconds
            """)
            
            if results['symbols_failed']:
                logger.warning(f"Failed symbols: {results['symbols_failed']}")
            
            return results['symbols_processed'] > 0
            
        except Exception as e:
            logger.error(f"Sentiment fetch failed: {e}")
            return False
    
    def run_daily_update(
        self,
        symbols: Optional[List[str]] = None
    ) -> bool:
        """
        Run daily sentiment update.
        
        Args:
            symbols: List of symbols (None = all)
        
        Returns:
            Success boolean
        """
        logger.info("Starting daily sentiment update")
        
        try:
            results = self.fetcher.update_latest_sentiment(
                symbols=symbols,
                deduplicate=True
            )
            
            logger.info(f"""
            Daily Sentiment Update Complete:
            - Symbols updated: {results['symbols_updated']}
            - New articles: {results['new_articles']:,}
            - Duplicates skipped: {results['duplicates_skipped']:,}
            - Duration: {results['duration']:.1f} seconds
            """)
            
            return results['symbols_updated'] > 0
            
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return False
    
    def run_sector_update(
        self,
        sector: Optional[str] = None
    ) -> bool:
        """
        Update sentiment for a specific sector.
        
        Args:
            sector: Sector name (None = all sectors)
        
        Returns:
            Success boolean
        """
        # Get symbols for sector
        if sector:
            if sector not in self.stocks_config:
                logger.error(f"Unknown sector: {sector}")
                return False
            symbols = [stock['ticker'] for stock in self.stocks_config[sector]]
            logger.info(f"Updating sentiment for sector {sector}: {len(symbols)} symbols")
        else:
            symbols = None
            logger.info("Updating sentiment for all sectors")
        
        return self.run_daily_update(symbols=symbols)
    
    def generate_sentiment_report(
        self,
        lookback_days: int = 30,
        output_file: Optional[str] = None
    ):
        """
        Generate sentiment analysis report.
        
        Args:
            lookback_days: Days to analyze
            output_file: Optional file to save report
        """
        logger.info(f"Generating sentiment report for last {lookback_days} days")
        
        try:
            # Get sentiment summary
            summary = self.fetcher.get_sentiment_summary()
            
            if summary.empty:
                logger.warning("No sentiment data found")
                return
            
            # Get daily aggregates
            daily_sentiment = self.fetcher.aggregate_daily_sentiment(
                lookback_days=lookback_days
            )
            
            # Generate report
            report = []
            report.append("=" * 80)
            report.append("SENTIMENT ANALYSIS REPORT")
            report.append(f"Generated: {datetime.now()}")
            report.append(f"Period: Last {lookback_days} days")
            report.append("=" * 80)
            
            # Overall statistics
            report.append("\nOVERALL STATISTICS:")
            report.append(f"Total symbols analyzed: {len(summary)}")
            report.append(f"Total articles: {summary['article_count'].sum():,}")
            report.append(f"Average sentiment: {summary['avg_sentiment'].mean():.3f}")
            
            # Sentiment distribution
            sentiment_dist = summary['sentiment_bias'].value_counts()
            report.append("\nSENTIMENT DISTRIBUTION:")
            for bias, count in sentiment_dist.items():
                report.append(f"  {bias}: {count} symbols ({count/len(summary)*100:.1f}%)")
            
            # Top bullish symbols
            report.append("\nTOP 10 BULLISH SYMBOLS:")
            top_bullish = summary.nlargest(10, 'avg_sentiment')[
                ['symbol', 'avg_sentiment', 'article_count']
            ]
            for _, row in top_bullish.iterrows():
                report.append(
                    f"  {row['symbol']}: {row['avg_sentiment']:.3f} "
                    f"({row['article_count']} articles)"
                )
            
            # Top bearish symbols
            report.append("\nTOP 10 BEARISH SYMBOLS:")
            top_bearish = summary.nsmallest(10, 'avg_sentiment')[
                ['symbol', 'avg_sentiment', 'article_count']
            ]
            for _, row in top_bearish.iterrows():
                report.append(
                    f"  {row['symbol']}: {row['avg_sentiment']:.3f} "
                    f"({row['article_count']} articles)"
                )
            
            # Most volatile sentiment
            report.append("\nMOST VOLATILE SENTIMENT (by std dev):")
            most_volatile = summary.nlargest(10, 'sentiment_volatility')[
                ['symbol', 'sentiment_volatility', 'avg_sentiment']
            ]
            for _, row in most_volatile.iterrows():
                report.append(
                    f"  {row['symbol']}: volatility={row['sentiment_volatility']:.3f}, "
                    f"avg={row['avg_sentiment']:.3f}"
                )
            
            # Recent trends
            if not daily_sentiment.empty:
                report.append("\nRECENT SENTIMENT TRENDS:")
                
                # Get last 7 days trends
                recent_date = datetime.now().date() - timedelta(days=7)
                recent_trends = daily_sentiment[daily_sentiment['date'] >= recent_date]
                
                if not recent_trends.empty:
                    trend_summary = recent_trends.groupby('symbol').agg({
                        'avg_sentiment': 'mean',
                        'sentiment_trend': 'last'
                    }).sort_values('sentiment_trend', ascending=False)
                    
                    report.append("\n  Improving Sentiment:")
                    improving = trend_summary.head(5)
                    for symbol, row in improving.iterrows():
                        report.append(
                            f"    {symbol}: trend={row['sentiment_trend']:.3f}, "
                            f"avg={row['avg_sentiment']:.3f}"
                        )
                    
                    report.append("\n  Declining Sentiment:")
                    declining = trend_summary.tail(5)
                    for symbol, row in declining.iterrows():
                        report.append(
                            f"    {symbol}: trend={row['sentiment_trend']:.3f}, "
                            f"avg={row['avg_sentiment']:.3f}"
                        )
            
            # Join report lines
            report_text = '\n'.join(report)
            
            # Display report
            logger.info(f"\n{report_text}")
            
            # Save to file if requested
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    def show_data_summary(self):
        """Display summary of stored sentiment data."""
        logger.info("Fetching sentiment data summary...")
        
        try:
            summary = self.fetcher.get_sentiment_summary()
            
            if summary.empty:
                logger.info("No sentiment data found")
                return
            
            # Display summary
            logger.info(f"\n{summary.to_string()}")
            
            # Overall stats
            total_articles = summary['article_count'].sum()
            total_symbols = len(summary)
            earliest_date = summary['first_date'].min()
            latest_date = summary['last_date'].max()
            
            logger.info(f"""
            Overall Statistics:
            - Total symbols: {total_symbols}
            - Total articles: {total_articles:,}
            - Date range: {earliest_date} to {latest_date}
            - Average sentiment: {summary['avg_sentiment'].mean():.3f}
            """)
            
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")


@click.group()
def cli():
    """Sentiment data ingestion tool for Parquet/GCS storage."""
    pass


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to fetch (can specify multiple)')
@click.option('--lookback', default=7, help='Days to look back')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def fetch(symbols, lookback, bucket):
    """Fetch sentiment data for symbols."""
    job = SentimentIngestionJob(bucket_name=bucket)
    
    # Convert symbols tuple to list
    symbols_list = list(symbols) if symbols else None
    
    success = job.run_sentiment_fetch(
        symbols=symbols_list,
        lookback_days=lookback
    )
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to update')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def update(symbols, bucket):
    """Update with latest sentiment data."""
    job = SentimentIngestionJob(bucket_name=bucket)
    
    symbols_list = list(symbols) if symbols else None
    
    success = job.run_daily_update(symbols=symbols_list)
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--sector', help='Sector to update (omit for all)')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def sector(sector, bucket):
    """Update sentiment for a specific sector."""
    job = SentimentIngestionJob(bucket_name=bucket)
    
    success = job.run_sector_update(sector=sector)
    
    sys.exit(0 if success else 1)


@cli.command()
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def summary(bucket):
    """Show summary of stored sentiment data."""
    job = SentimentIngestionJob(bucket_name=bucket)
    job.show_data_summary()


@cli.command()
@click.option('--lookback', default=30, help='Days to analyze')
@click.option('--output', help='Output file for report')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def report(lookback, output, bucket):
    """Generate sentiment analysis report."""
    job = SentimentIngestionJob(bucket_name=bucket)
    job.generate_sentiment_report(
        lookback_days=lookback,
        output_file=output
    )


@cli.command()
@click.option('--interval', default=6, help='Hours between updates')
@click.option('--bucket', default=GCS_BUCKET, help='GCS bucket name')
def continuous(interval, bucket):
    """Run continuous sentiment updates."""
    job = SentimentIngestionJob(bucket_name=bucket)
    
    logger.info(f"Starting continuous sentiment updates (every {interval} hours)")
    
    while True:
        try:
            # Run update
            logger.info(f"Running update at {datetime.now()}")
            job.run_daily_update()
            
            # Wait for next run
            next_run = datetime.now() + timedelta(hours=interval)
            wait_seconds = (next_run - datetime.now()).total_seconds()
            logger.info(f"Next update at {next_run} ({wait_seconds:.0f}s)")
            
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("Continuous updates stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous update: {e}")
            logger.info("Waiting 30 minutes before retry...")
            time.sleep(1800)


if __name__ == '__main__':
    cli()