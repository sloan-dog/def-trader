"""
Model training job for scheduled training runs.
"""
import click
import sys
from datetime import datetime, timedelta
from loguru import logger
import mlflow
import json

from src.training.trainer import GNNTrainer
from src.training.prediction_pipeline import PredictionPipeline
from src.training.metrics import ModelMetrics
from src.utils.bigquery_client import BigQueryClient
from config.settings import MODEL_CONFIG, BQ_TABLES


class TrainingJob:
    """Orchestrate model training and evaluation."""

    def __init__(self, experiment_name: str = "temporal_gnn_trading"):
        """Initialize training job."""
        self.trainer = GNNTrainer(experiment_name=experiment_name)
        self.bq_client = BigQueryClient()
        self.metrics_calculator = ModelMetrics()

        # Configure logging
        logger.add(
            "logs/training_{time}.log",
            rotation="1 week",
            retention="4 weeks",
            level="INFO"
        )

    def run(
            self,
            end_date: str = None,
            lookback_months: int = 24,
            validate_only: bool = False
    ):
        """Run training job."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        start_date = (
                datetime.strptime(end_date, '%Y-%m-%d') -
                timedelta(days=lookback_months * 30)
        ).strftime('%Y-%m-%d')

        logger.info(f"Starting training job from {start_date} to {end_date}")

        results = {
            'job_type': 'model_training',
            'start_date': start_date,
            'end_date': end_date,
            'start_time': datetime.now(),
            'mlflow_run_id': None
        }

        try:
            # Check data availability
            data_check = self._check_data_availability(start_date, end_date)
            results['data_check'] = data_check

            if not data_check['sufficient_data']:
                raise ValueError(f"Insufficient data: {data_check['issues']}")

            if validate_only:
                # Only run validation on existing model
                results['validation_results'] = self._validate_existing_model(end_date)
            else:
                # Run full training
                training_results = self.trainer.run_full_training_pipeline(
                    start_date=start_date,
                    end_date=end_date
                )

                results['training_results'] = {
                    'val_metrics': training_results['val_metrics'],
                    'test_metrics': training_results['test_metrics']
                }

                # Get MLflow run ID
                results['mlflow_run_id'] = mlflow.active_run().info.run_id if mlflow.active_run() else None

                # Deploy model if performance is good
                if self._should_deploy_model(training_results['test_metrics']):
                    deploy_success = self._deploy_model(training_results['model'])
                    results['deployment'] = {
                        'deployed': deploy_success,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    results['deployment'] = {
                        'deployed': False,
                        'reason': 'Performance below threshold'
                    }

            results['success'] = True

        except Exception as e:
            logger.error(f"Training job failed: {e}")
            results['success'] = False
            results['error'] = str(e)

        finally:
            results['end_time'] = datetime.now()
            results['duration_hours'] = (
                    (results['end_time'] - results['start_time']).total_seconds() / 3600
            )

            # Log results
            self._log_job_results(results)

        return results

    def _check_data_availability(self, start_date: str, end_date: str) -> Dict:
        """Check if sufficient data is available for training."""
        checks = {
            'sufficient_data': True,
            'issues': [],
            'statistics': {}
        }

        # Check OHLCV data
        query = f"""
        SELECT 
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT date) as unique_dates,
            COUNT(*) as total_records,
            MIN(date) as min_date,
            MAX(date) as max_date
        FROM `{BQ_TABLES['raw_ohlcv']}`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """

        ohlcv_stats = self.bq_client.query(query).iloc[0]
        checks['statistics']['ohlcv'] = ohlcv_stats.to_dict()

        # Minimum requirements
        min_tickers = 20
        min_dates = 250  # ~1 year of trading days

        if ohlcv_stats['unique_tickers'] < min_tickers:
            checks['sufficient_data'] = False
            checks['issues'].append(
                f"Only {ohlcv_stats['unique_tickers']} tickers available, need {min_tickers}"
            )

        if ohlcv_stats['unique_dates'] < min_dates:
            checks['sufficient_data'] = False
            checks['issues'].append(
                f"Only {ohlcv_stats['unique_dates']} dates available, need {min_dates}"
            )

        # Check technical indicators coverage
        query = f"""
        SELECT 
            COUNT(DISTINCT ticker) as tickers_with_indicators,
            AVG(CASE WHEN rsi IS NOT NULL THEN 1 ELSE 0 END) as rsi_coverage,
            AVG(CASE WHEN macd IS NOT NULL THEN 1 ELSE 0 END) as macd_coverage
        FROM `{BQ_TABLES['technical_indicators']}`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """

        indicator_stats = self.bq_client.query(query).iloc[0]
        checks['statistics']['indicators'] = indicator_stats.to_dict()

        if indicator_stats['rsi_coverage'] < 0.8:
            checks['issues'].append(
                f"RSI coverage only {indicator_stats['rsi_coverage']:.1%}"
            )

        return checks

    def _validate_existing_model(self, evaluation_date: str) -> Dict:
        """Validate existing model performance."""
        # Get latest model version
        query = f"""
        SELECT model_version, model_path
        FROM `{BQ_TABLES['model_metadata']}`
        ORDER BY created_at DESC
        LIMIT 1
        """

        model_info = self.bq_client.query(query)

        if model_info.empty:
            return {'error': 'No trained model found'}

        model_version = model_info['model_version'].iloc[0]
        model_path = model_info['model_path'].iloc[0]

        # Initialize prediction pipeline
        pipeline = PredictionPipeline(model_path=model_path)

        # Generate predictions for recent dates
        lookback_days = 30
        start_date = (
                datetime.strptime(evaluation_date, '%Y-%m-%d') -
                timedelta(days=lookback_days)
        ).strftime('%Y-%m-%d')

        predictions = pipeline.generate_batch_predictions(
            start_date=start_date,
            end_date=evaluation_date
        )

        # Evaluate predictions
        evaluation = pipeline.evaluate_historical_predictions(
            start_date=start_date,
            end_date=evaluation_date
        )

        # Calculate metrics
        metrics = {}
        for horizon in ['1d', '7d', '30d', '60d']:
            pred_col = f'horizon_{horizon}'
            actual_col = f'actual_{horizon}'

            if pred_col in evaluation.columns and actual_col in evaluation.columns:
                horizon_metrics = self.metrics_calculator.calculate_metrics(
                    evaluation[pred_col].values,
                    evaluation[actual_col].values
                )

                for metric_name, value in horizon_metrics.items():
                    metrics[f'{horizon}_{metric_name}'] = value

        return {
            'model_version': model_version,
            'evaluation_period': f'{start_date} to {evaluation_date}',
            'metrics': metrics
        }

    def _should_deploy_model(self, test_metrics: Dict) -> bool:
        """Determine if model should be deployed based on metrics."""
        # Define minimum performance thresholds
        thresholds = {
            'direction_accuracy': 0.52,  # Better than random
            'sharpe_ratio': 0.5,  # Minimum risk-adjusted return
            'max_drawdown': -0.2  # Maximum acceptable drawdown
        }

        # Check each horizon
        horizons_passed = 0
        for horizon in ['1d', '7d', '30d', '60d']:
            horizon_pass = True

            for metric, threshold in thresholds.items():
                metric_key = f'{horizon}_{metric}'
                if metric_key in test_metrics:
                    value = test_metrics[metric_key]

                    if metric == 'max_drawdown':
                        # Drawdown is negative, so check if it's not worse than threshold
                        if value < threshold:
                            horizon_pass = False
                            break
                    else:
                        # Other metrics should be above threshold
                        if value < threshold:
                            horizon_pass = False
                            break

            if horizon_pass:
                horizons_passed += 1

        # Require at least 2 horizons to pass
        return horizons_passed >= 2

    def _deploy_model(self, model) -> bool:
        """Deploy model to production."""
        try:
            # Save model to production location
            production_path = f"models/production/tgnn_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

            # In real implementation, this would upload to GCS
            import torch
            torch.save({
                'model_state_dict': model.state_dict(),
                'deployment_time': datetime.now().isoformat(),
                'model_config': MODEL_CONFIG
            }, production_path)

            logger.info(f"Model deployed to {production_path}")
            return True

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False

    def _log_job_results(self, results: Dict):
        """Log job results to BigQuery."""
        try:
            job_log = {
                'job_name': 'model_training',
                'run_date': datetime.now().date(),
                'start_time': results['start_time'],
                'end_time': results['end_time'],
                'duration_hours': results['duration_hours'],
                'success': results['success'],
                'mlflow_run_id': results.get('mlflow_run_id'),
                'metrics': json.dumps(results.get('training_results', {})),
                'deployment_status': json.dumps(results.get('deployment', {})),
                'error': results.get('error')
            }

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
@click.option('--end-date', default=None, help='End date for training data (YYYY-MM-DD)')
@click.option('--lookback-months', default=24, help='Months of historical data to use')
@click.option('--validate-only', is_flag=True, help='Only validate existing model')
@click.option('--experiment-name', default='temporal_gnn_trading', help='MLflow experiment name')
def main(end_date, lookback_months, validate_only, experiment_name):
    """Run model training job."""
    job = TrainingJob(experiment_name=experiment_name)
    results = job.run(
        end_date=end_date,
        lookback_months=lookback_months,
        validate_only=validate_only
    )

    if results['success']:
        logger.info("Training job completed successfully")
        sys.exit(0)
    else:
        logger.error("Training job failed")
        sys.exit(1)


if __name__ == '__main__':
    main()