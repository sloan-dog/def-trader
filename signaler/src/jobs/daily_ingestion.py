"""
Daily data ingestion job for all data sources.
"""
import click
import sys
from datetime import datetime, timedelta
from typing import Dict
from loguru import logger
import traceback

from src.shared_logging import setup_logging, log_exception
from src.data_ingestion.ohlcv_fetcher import OHLCVUpdater
from src.data_ingestion.macro_data_fetcher import MacroDataUpdater
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from src.feature_engineering.technical_indicators import TechnicalIndicatorCalculator
from src.feature_engineering.temporal_features import TemporalFeatureEngineer
from src.feature_engineering.graph_constructor import StockGraphConstructor
from src.utils.bigquery import BigQueryClient
from config.settings import BQ_TABLES, load_stocks_config


class DailyIngestionJob:
    """Orchestrate daily data ingestion."""

    def __init__(self):
        """Initialize job components."""
        self.is_cloud_run = self._is_cloud_run()
        
        # Configure logging with JSON format for Google Cloud
        setup_logging(
            level="INFO",
            log_file="logs/daily_ingestion_{time}.log" if not self.is_cloud_run else None,
            rotation="1 day",
            retention="30 days",
            app_name="signaler-daily-ingestion"
        )
        
        self.ohlcv_updater = OHLCVUpdater()
        self.macro_updater = MacroDataUpdater()
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.bq_client = BigQueryClient()
        self.stocks_config = load_stocks_config()
    
    def _is_cloud_run(self) -> bool:
        """Check if running in Google Cloud Run."""
        import os
        return bool(os.environ.get("K_SERVICE") or os.environ.get("GOOGLE_CLOUD_PROJECT"))

    def run(self, date: str = None):
        """Run daily ingestion job."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Starting daily ingestion for {date}")

        results = {
            'date': date,
            'start_time': datetime.now(),
            'steps': {}
        }

        try:
            # Step 1: Update OHLCV data
            logger.info("Step 1: Updating OHLCV data")
            ohlcv_success = self._update_ohlcv_data()
            results['steps']['ohlcv'] = {'success': ohlcv_success}

            # Step 2: Calculate technical indicators
            logger.info("Step 2: Calculating technical indicators")
            indicators_success = self._calculate_indicators(date)
            results['steps']['indicators'] = {'success': indicators_success}

            # Step 3: Update macro data
            logger.info("Step 3: Updating macro data")
            macro_success = self._update_macro_data()
            results['steps']['macro'] = {'success': macro_success}

            # Step 4: Update sentiment data
            logger.info("Step 4: Updating sentiment data")
            sentiment_success = self._update_sentiment_data(date)
            results['steps']['sentiment'] = {'success': sentiment_success}

            # Step 5: Generate temporal features
            logger.info("Step 5: Generating temporal features")
            temporal_success = self._generate_temporal_features(date)
            results['steps']['temporal'] = {'success': temporal_success}

            # Step 6: Update feature view
            logger.info("Step 6: Updating feature view")
            view_success = self._update_feature_view()
            results['steps']['feature_view'] = {'success': view_success}

            # Step 7: Data quality checks
            logger.info("Step 7: Running data quality checks")
            quality_results = self._run_quality_checks(date)
            results['steps']['quality_checks'] = quality_results

            # Calculate overall success
            results['overall_success'] = all(
                step.get('success', False)
                for step in results['steps'].values()
                if isinstance(step, dict)
            )

        except Exception as e:
            log_exception("Daily ingestion failed", exception=e)
            results['overall_success'] = False
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()

        finally:
            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()

            # Log results
            self._log_job_results(results)

        return results

    def _update_ohlcv_data(self) -> bool:
        """Update OHLCV data for all tickers."""
        try:
            self.ohlcv_updater.run_daily_update()
            return True
        except Exception as e:
            log_exception("OHLCV update failed", exception=e)
            return False

    def _calculate_indicators(self, date: str) -> bool:
        """Calculate technical indicators for updated data."""
        try:
            # Get tickers that were updated today
            query = f"""
            SELECT DISTINCT ticker
            FROM `{BQ_TABLES['raw_ohlcv']}`
            WHERE DATE(inserted_at) = CURRENT_DATE()
            """

            updated_tickers = self.bq_client.query(query)

            if updated_tickers.empty:
                logger.warning("No tickers were updated today")
                return True

            success_count = 0

            for ticker in updated_tickers['ticker']:
                try:
                    # Get recent OHLCV data
                    lookback_days = 200  # Need history for indicators
                    start_date = (
                            datetime.strptime(date, '%Y-%m-%d') - timedelta(days=lookback_days)
                    ).strftime('%Y-%m-%d')

                    query = f"""
                    SELECT *
                    FROM `{BQ_TABLES['raw_ohlcv']}`
                    WHERE ticker = '{ticker}'
                      AND date >= '{start_date}'
                    ORDER BY date
                    """

                    ohlcv_df = self.bq_client.query(query)

                    if len(ohlcv_df) < 20:  # Minimum data needed
                        logger.warning(f"Insufficient data for {ticker}")
                        continue

                    # Calculate indicators
                    indicators_df = self.indicator_calculator.calculate_all_indicators(
                        ohlcv_df,
                        ticker=ticker
                    )

                    # Get only new indicators (last few days)
                    recent_indicators = indicators_df.tail(5)

                    # Store to BigQuery
                    self.bq_client.insert_dataframe(
                        recent_indicators,
                        'technical_indicators',
                        if_exists='append'
                    )

                    success_count += 1

                except Exception as e:
                    log_exception(f"Failed to calculate indicators for {ticker}", exception=e, ticker=ticker)

            logger.info(f"Calculated indicators for {success_count}/{len(updated_tickers)} tickers")
            return success_count > 0

        except Exception as e:
            log_exception("Indicator calculation failed", exception=e)
            return False

    def _update_macro_data(self) -> bool:
        """Update macro economic data."""
        try:
            return self.macro_updater.run_full_update()
        except Exception as e:
            logger.error(f"Macro data update failed: {e}")
            return False

    def _update_sentiment_data(self, date: str) -> bool:
        """Update sentiment data."""
        try:
            av_client = AlphaVantageClient()

            # Get all tickers
            all_tickers = []
            for sector, stocks in self.stocks_config.items():
                for stock in stocks:
                    all_tickers.append(stock['ticker'])

            # Fetch sentiment data
            sentiment_df = av_client.get_sentiment_data(all_tickers)

            if not sentiment_df.empty:
                # Filter to recent data
                sentiment_df = sentiment_df[
                    sentiment_df['date'] >= (
                            datetime.strptime(date, '%Y-%m-%d') - timedelta(days=7)
                    ).date()
                    ]

                # Add sector information
                ticker_to_sector = {}
                for sector, stocks in self.stocks_config.items():
                    for stock in stocks:
                        ticker_to_sector[stock['ticker']] = sector

                sentiment_df['sector'] = sentiment_df['ticker'].map(ticker_to_sector)

                # Store to BigQuery
                self.bq_client.insert_dataframe(
                    sentiment_df,
                    'sentiment_data',
                    if_exists='append'
                )

                logger.info(f"Updated sentiment data for {len(sentiment_df)} records")
                return True

            return False

        except Exception as e:
            logger.error(f"Sentiment update failed: {e}")
            return False

    def _generate_temporal_features(self, date: str) -> bool:
        """Generate temporal features."""
        try:
            # Generate features for next 30 days
            end_date = (
                    datetime.strptime(date, '%Y-%m-%d') + timedelta(days=30)
            ).strftime('%Y-%m-%d')

            return self.temporal_engineer.store_temporal_features(date, end_date)

        except Exception as e:
            logger.error(f"Temporal feature generation failed: {e}")
            return False

    def _update_feature_view(self) -> bool:
        """Update the materialized feature view."""
        try:
            self.bq_client.create_feature_view()
            return True
        except Exception as e:
            logger.error(f"Feature view update failed: {e}")
            return False

    def _run_quality_checks(self, date: str) -> Dict:
        """Run data quality checks."""
        quality_results = {
            'checks_passed': 0,
            'checks_failed': 0,
            'issues': []
        }

        try:
            # Check 1: Data completeness
            query = f"""
            SELECT 
                COUNT(DISTINCT ticker) as unique_tickers,
                COUNT(*) as total_records,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM `{BQ_TABLES['raw_ohlcv']}`
            WHERE DATE(inserted_at) = CURRENT_DATE()
            """

            completeness = self.bq_client.query(query)

            if completeness['unique_tickers'].iloc[0] < 10:
                quality_results['issues'].append("Less than 10 tickers updated today")
                quality_results['checks_failed'] += 1
            else:
                quality_results['checks_passed'] += 1

            # Check 2: Data freshness
            if completeness['max_date'].iloc[0] < datetime.now().date() - timedelta(days=2):
                quality_results['issues'].append("Data is more than 2 days old")
                quality_results['checks_failed'] += 1
            else:
                quality_results['checks_passed'] += 1

            # Check 3: Indicator coverage
            query = f"""
            SELECT 
                COUNT(DISTINCT o.ticker) as tickers_with_ohlcv,
                COUNT(DISTINCT t.ticker) as tickers_with_indicators
            FROM `{BQ_TABLES['raw_ohlcv']}` o
            LEFT JOIN `{BQ_TABLES['technical_indicators']}` t
            ON o.ticker = t.ticker AND o.date = t.date
            WHERE o.date = '{date}'
            """

            coverage = self.bq_client.query(query)

            coverage_ratio = (
                    coverage['tickers_with_indicators'].iloc[0] /
                    coverage['tickers_with_ohlcv'].iloc[0]
            )

            if coverage_ratio < 0.9:
                quality_results['issues'].append(
                    f"Only {coverage_ratio:.1%} of tickers have indicators"
                )
                quality_results['checks_failed'] += 1
            else:
                quality_results['checks_passed'] += 1

        except Exception as e:
            logger.error(f"Quality checks failed: {e}")
            quality_results['error'] = str(e)

        quality_results['success'] = quality_results['checks_failed'] == 0

        return quality_results

    def _log_job_results(self, results: Dict):
        """Log job results to BigQuery."""
        try:
            job_log = {
                'job_name': 'daily_ingestion',
                'run_date': results['date'],
                'start_time': results['start_time'],
                'end_time': results['end_time'],
                'duration_seconds': results['duration'],
                'overall_success': results['overall_success'],
                'step_results': str(results['steps']),
                'error': results.get('error', None)
            }

            # Store to job logs table
            import pandas as pd
            log_df = pd.DataFrame([job_log])

            self.bq_client.insert_dataframe(
                log_df,
                'job_logs',
                if_exists='append'
            )

        except Exception as e:
            logger.error(f"Failed to log job results: {e}")


@click.command()
@click.option('--date', default=None, help='Date to run ingestion for (YYYY-MM-DD)')
def main(date):
    """Run daily data ingestion job."""
    try:
        job = DailyIngestionJob()
        results = job.run(date)

        if results['overall_success']:
            logger.info("Daily ingestion completed successfully", 
                       duration=results.get('duration', 0),
                       steps_completed=len(results.get('steps', {})))
            sys.exit(0)
        else:
            logger.error("Daily ingestion failed - see logs for details",
                        duration=results.get('duration', 0),
                        error=results.get('error', 'Unknown error'),
                        steps=results.get('steps', {}))
            sys.exit(1)
    except Exception as e:
        log_exception("Fatal error in daily ingestion job", exception=e)
        sys.exit(1)


if __name__ == '__main__':
    main()