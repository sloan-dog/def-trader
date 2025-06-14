"""
Prediction pipeline for generating trading signals.
"""
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
from loguru import logger

from config.settings import BQ_TABLES, MODEL_CONFIG
from src.models.temporal_gnn import TemporalGNN, StockGraphDataset
from src.feature_engineering.graph_constructor import StockGraphConstructor
from src.utils.bigquery_client import BigQueryClient


class PredictionPipeline:
    """Generate predictions using trained model."""

    def __init__(
            self,
            model_path: str = None,
            model_version: str = None
    ):
        """Initialize prediction pipeline."""
        self.bq_client = BigQueryClient()
        self.graph_constructor = StockGraphConstructor()

        # Load model
        if model_path:
            self.load_model(model_path)
        elif model_version:
            self.load_model_from_registry(model_version)
        else:
            raise ValueError("Either model_path or model_version must be provided")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: str):
        """Load model from file."""
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract configuration
        self.model_config = checkpoint['model_config']
        self.feature_columns = checkpoint['feature_columns']

        # Initialize model
        self.model = TemporalGNN(
            num_node_features=self.model_config['num_node_features'],
            num_edge_features=self.model_config['num_edge_features'],
            hidden_dim=self.model_config['hidden_dim'],
            prediction_horizons=self.model_config['prediction_horizons']
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load feature scaler
        scaler_path = Path(model_path).parent / 'feature_scaler.pkl'
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
        else:
            logger.warning("Feature scaler not found, predictions may be uncalibrated")
            self.feature_scaler = None

        logger.info(f"Loaded model from {model_path}")

    def load_model_from_registry(self, model_version: str):
        """Load model from BigQuery registry."""
        query = f"""
        SELECT *
        FROM `{BQ_TABLES['model_metadata']}`
        WHERE model_version = '{model_version}'
        """

        metadata = self.bq_client.query(query)
        if metadata.empty:
            raise ValueError(f"Model version {model_version} not found")

        model_path = metadata['model_path'].iloc[0]
        # Download from GCS and load
        # This would require GCS integration
        raise NotImplementedError("GCS model loading not implemented")

    def prepare_features(
            self,
            prediction_date: str,
            lookback_days: int = 90
    ) -> Tuple[pd.DataFrame, Dict]:
        """Prepare features for prediction."""
        start_date = (
                pd.to_datetime(prediction_date) - timedelta(days=lookback_days + 30)
        ).strftime('%Y-%m-%d')

        # Query features
        query = f"""
        SELECT *
        FROM `{self.bq_client.dataset_ref}.feature_view`
        WHERE date BETWEEN '{start_date}' AND '{prediction_date}'
        ORDER BY ticker, date
        """

        features_df = self.bq_client.query(query)

        # Build graph
        graph_structure = self.graph_constructor.build_graph(
            start_date=start_date,
            end_date=prediction_date
        )

        # Scale features
        if self.feature_scaler:
            features_df[self.feature_columns] = self.feature_scaler.transform(
                features_df[self.feature_columns]
            )

        return features_df, graph_structure

    def generate_predictions(
            self,
            prediction_date: str,
            tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate predictions for specified date."""
        # Prepare features
        features_df, graph_structure = self.prepare_features(prediction_date)

        # Filter tickers if specified
        if tickers:
            features_df = features_df[features_df['ticker'].isin(tickers)]

        # Create dataset
        dataset = StockGraphDataset(
            features_df,
            graph_structure,
            sequence_length=MODEL_CONFIG['historical_window'],
            prediction_horizons=MODEL_CONFIG['prediction_horizons']
        )

        # Generate predictions
        batch = dataset.create_graph_batch(
            pd.to_datetime(prediction_date),
            lookback_days=MODEL_CONFIG['historical_window']
        )

        if batch is None:
            logger.error(f"Could not create batch for {prediction_date}")
            return pd.DataFrame()

        batch = batch.to(self.device)

        with torch.no_grad():
            predictions = self.model(
                batch.x,
                batch.edge_index,
                batch.temporal_sequences,
                batch.batch if hasattr(batch, 'batch') else None
            )

        # Convert to DataFrame
        results = []

        # Get ticker list from dataset
        valid_tickers = [t for t in dataset.tickers if t in features_df['ticker'].unique()]

        for i, ticker in enumerate(valid_tickers):
            result = {
                'ticker': ticker,
                'prediction_date': prediction_date,
                'model_version': self.model_config.get('version', 'unknown')
            }

            for horizon in MODEL_CONFIG['prediction_horizons']:
                horizon_key = f'horizon_{horizon}d'
                pred, conf = predictions[horizon_key]

                result[f'horizon_{horizon}d'] = pred[i].item()
                result[f'confidence_{horizon}d'] = conf[i].item()

            results.append(result)

        return pd.DataFrame(results)

    def generate_batch_predictions(
            self,
            start_date: str,
            end_date: str,
            tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate predictions for date range."""
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        all_predictions = []

        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Generating predictions for {date_str}")

            try:
                predictions = self.generate_predictions(date_str, tickers)
                all_predictions.append(predictions)
            except Exception as e:
                logger.error(f"Failed to generate predictions for {date_str}: {e}")

        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        return pd.DataFrame()

    def store_predictions(
            self,
            predictions_df: pd.DataFrame,
            prediction_id: str = None
    ) -> bool:
        """Store predictions to BigQuery."""
        if prediction_id is None:
            prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        predictions_df['prediction_id'] = prediction_id
        predictions_df['created_at'] = datetime.now()

        try:
            self.bq_client.insert_dataframe(
                predictions_df,
                'predictions',
                if_exists='append'
            )
            logger.info(f"Stored {len(predictions_df)} predictions")
            return True
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            return False

    def evaluate_historical_predictions(
            self,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """Evaluate historical prediction accuracy."""
        query = f"""
        WITH predictions AS (
            SELECT 
                ticker,
                prediction_date,
                horizon_1d,
                horizon_7d,
                horizon_30d,
                horizon_60d
            FROM `{BQ_TABLES['predictions']}`
            WHERE prediction_date BETWEEN '{start_date}' AND '{end_date}'
        ),
        actuals AS (
            SELECT 
                ticker,
                date,
                close as current_price,
                LEAD(close, 1) OVER (PARTITION BY ticker ORDER BY date) as price_1d,
                LEAD(close, 7) OVER (PARTITION BY ticker ORDER BY date) as price_7d,
                LEAD(close, 30) OVER (PARTITION BY ticker ORDER BY date) as price_30d,
                LEAD(close, 60) OVER (PARTITION BY ticker ORDER BY date) as price_60d
            FROM `{BQ_TABLES['raw_ohlcv']}`
        )
        SELECT 
            p.*,
            a.current_price,
            (a.price_1d - a.current_price) / a.current_price as actual_1d,
            (a.price_7d - a.current_price) / a.current_price as actual_7d,
            (a.price_30d - a.current_price) / a.current_price as actual_30d,
            (a.price_60d - a.current_price) / a.current_price as actual_60d
        FROM predictions p
        JOIN actuals a
        ON p.ticker = a.ticker AND p.prediction_date = a.date
        """

        return self.bq_client.query(query)

    def generate_trading_signals(
            self,
            predictions_df: pd.DataFrame,
            confidence_threshold: float = 0.6,
            return_threshold: float = 0.02
    ) -> pd.DataFrame:
        """Convert predictions to actionable trading signals."""
        signals = []

        for _, row in predictions_df.iterrows():
            # Check each horizon
            for horizon in [1, 7, 30, 60]:
                pred_col = f'horizon_{horizon}d'
                conf_col = f'confidence_{horizon}d'

                if pred_col in row and conf_col in row:
                    prediction = row[pred_col]
                    confidence = row[conf_col]

                    # Generate signal if confidence and return thresholds met
                    if confidence >= confidence_threshold and abs(prediction) >= return_threshold:
                        signal = {
                            'ticker': row['ticker'],
                            'date': row['prediction_date'],
                            'horizon': f'{horizon}d',
                            'signal': 'BUY' if prediction > 0 else 'SELL',
                            'predicted_return': prediction,
                            'confidence': confidence,
                            'strength': abs(prediction) * confidence
                        }
                        signals.append(signal)

        signals_df = pd.DataFrame(signals)

        # Rank signals by strength
        if not signals_df.empty:
            signals_df['rank'] = signals_df.groupby(['date', 'horizon'])['strength'].rank(
                ascending=False,
                method='dense'
            )

        return signals_df

    def create_portfolio_allocation(
            self,
            signals_df: pd.DataFrame,
            max_positions: int = 20,
            risk_parity: bool = True
    ) -> pd.DataFrame:
        """Create portfolio allocation from signals."""
        allocations = []

        # Group by date and horizon
        for (date, horizon), group in signals_df.groupby(['date', 'horizon']):
            # Select top signals
            top_signals = group.nsmallest(max_positions, 'rank')

            if risk_parity:
                # Calculate risk-based weights
                total_strength = top_signals['strength'].sum()
                weights = top_signals['strength'] / total_strength
            else:
                # Equal weight
                weights = pd.Series([1.0 / len(top_signals)] * len(top_signals))

            for (_, signal), weight in zip(top_signals.iterrows(), weights):
                allocation = {
                    'date': date,
                    'horizon': horizon,
                    'ticker': signal['ticker'],
                    'signal': signal['signal'],
                    'weight': weight * (1 if signal['signal'] == 'BUY' else -1),
                    'predicted_return': signal['predicted_return'],
                    'confidence': signal['confidence']
                }
                allocations.append(allocation)

        return pd.DataFrame(allocations)


class RealTimePredictionService:
    """Service for real-time prediction generation."""

    def __init__(self, model_version: str):
        """Initialize real-time service."""
        self.pipeline = PredictionPipeline(model_version=model_version)
        self.last_update = None

    def get_latest_predictions(
            self,
            tickers: List[str],
            force_update: bool = False
    ) -> pd.DataFrame:
        """Get latest predictions, updating if necessary."""
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Check if update needed
        if not force_update and self.last_update == current_date:
            # Return cached predictions
            return self._get_cached_predictions(tickers)

        # Generate new predictions
        predictions = self.pipeline.generate_predictions(
            current_date,
            tickers
        )

        # Store predictions
        self.pipeline.store_predictions(predictions)

        # Update cache
        self.last_update = current_date

        return predictions

    def _get_cached_predictions(self, tickers: List[str]) -> pd.DataFrame:
        """Get cached predictions from BigQuery."""
        ticker_list = "','".join(tickers)

        query = f"""
        SELECT *
        FROM `{BQ_TABLES['predictions']}`
        WHERE prediction_date = CURRENT_DATE()
          AND ticker IN ('{ticker_list}')
        """

        return self.pipeline.bq_client.query(query)

    def stream_predictions(
            self,
            tickers: List[str],
            update_interval: int = 3600
    ):
        """Stream predictions with periodic updates."""
        import time

        while True:
            try:
                # Get latest predictions
                predictions = self.get_latest_predictions(tickers)

                # Generate signals
                signals = self.pipeline.generate_trading_signals(predictions)

                # Yield results
                yield {
                    'timestamp': datetime.now(),
                    'predictions': predictions,
                    'signals': signals
                }

                # Wait for next update
                time.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in prediction stream: {e}")
                time.sleep(60)  # Wait before retry