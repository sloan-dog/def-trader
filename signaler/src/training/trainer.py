"""
Training pipeline for Temporal GNN model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from src.utils import logger
import mlflow
from sklearn.preprocessing import RobustScaler
from google.cloud import aiplatform

from config.settings import (
    MODEL_CONFIG,
    GNN_CONFIG,
    VALIDATION_CONFIG,
    BQ_TABLES,
    VERTEX_AI_CONFIG
)
from src.models.temporal_gnn import TemporalGNN, StockGraphDataset
from src.feature_engineering.graph_constructor import StockGraphConstructor
from src.utils.bigquery import BigQueryClient
from src.training.metrics import ModelMetrics


class GNNTrainer:
    """Trainer for Temporal GNN model."""

    def __init__(
            self,
            experiment_name: str = "temporal_gnn_trading",
            mlflow_uri: str = None,
            use_vertex_ai: bool = True
    ):
        """Initialize trainer."""
        self.bq_client = BigQueryClient()
        self.graph_constructor = StockGraphConstructor()
        self.metrics_calculator = ModelMetrics()

        # Model configuration
        self.model_config = MODEL_CONFIG
        self.gnn_config = GNN_CONFIG
        self.validation_config = VALIDATION_CONFIG

        # MLflow setup
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        # Vertex AI setup
        self.use_vertex_ai = use_vertex_ai
        if use_vertex_ai:
            try:
                aiplatform.init(
                    project=VERTEX_AI_CONFIG['project'],
                    location=VERTEX_AI_CONFIG['location'],
                    experiment=VERTEX_AI_CONFIG['experiment'],
                    experiment_description=VERTEX_AI_CONFIG['experiment_description']
                )
                logger.info("Vertex AI Metadata Store initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Vertex AI: {e}")
                self.use_vertex_ai = False

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def prepare_training_data(
            self,
            start_date: str,
            end_date: str,
            feature_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Prepare data for training."""
        logger.info(f"Preparing data from {start_date} to {end_date}")

        # Query feature data
        query = f"""
        SELECT *
        FROM `{self.bq_client.dataset_ref}.feature_view`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """

        features_df = self.bq_client.query(query)
        logger.info(f"Loaded {len(features_df)} feature records")

        # Build graph structure
        graph_structure = self.graph_constructor.build_graph(
            start_date=start_date,
            end_date=end_date
        )

        # Select feature columns
        if feature_columns is None:
            # Default feature selection
            exclude_cols = [
                'ticker', 'date', 'inserted_at', 'sector',
                'industry', 'market_cap_category'
            ]
            feature_columns = [
                col for col in features_df.columns
                if col not in exclude_cols
            ]

        # Handle missing values
        features_df[feature_columns] = features_df[feature_columns].fillna(
            features_df[feature_columns].rolling(window=5, min_periods=1).mean()
        )

        # Scale features
        scaler = RobustScaler()
        features_df[feature_columns] = scaler.fit_transform(features_df[feature_columns])

        # Save scaler for inference
        self.feature_scaler = scaler
        self.feature_columns = feature_columns

        return features_df, graph_structure

    def create_train_val_test_split(
            self,
            features_df: pd.DataFrame,
            test_size: float = 0.15,
            val_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/validation/test split."""
        # Sort by date
        features_df = features_df.sort_values('date')

        # Get unique dates
        unique_dates = features_df['date'].unique()
        n_dates = len(unique_dates)

        # Calculate split indices
        train_end_idx = int(n_dates * (1 - test_size - val_size))
        val_end_idx = int(n_dates * (1 - test_size))

        train_end_date = unique_dates[train_end_idx]
        val_end_date = unique_dates[val_end_idx]

        # Split data
        train_df = features_df[features_df['date'] <= train_end_date]
        val_df = features_df[
            (features_df['date'] > train_end_date) &
            (features_df['date'] <= val_end_date)
            ]
        test_df = features_df[features_df['date'] > val_end_date]

        logger.info(f"Train: {len(train_df)} records up to {train_end_date}")
        logger.info(f"Val: {len(val_df)} records up to {val_end_date}")
        logger.info(f"Test: {len(test_df)} records from {test_df['date'].min()}")

        return train_df, val_df, test_df

    def initialize_model(
            self,
            num_node_features: int,
            num_edge_features: int = 0
    ) -> TemporalGNN:
        """Initialize the model."""
        model = TemporalGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=self.gnn_config['hidden_dim'],
            num_gnn_layers=self.gnn_config['num_gnn_layers'],
            num_temporal_layers=self.gnn_config['num_temporal_layers'],
            num_heads=self.gnn_config['attention_heads'],
            dropout=self.gnn_config['dropout_rate'],
            prediction_horizons=self.model_config['prediction_horizons']
        )

        return model.to(self.device)

    def train_epoch(
            self,
            model: TemporalGNN,
            dataset: StockGraphDataset,
            optimizer: torch.optim.Optimizer,
            dates: List[pd.Timestamp]
    ) -> Dict[str, float]:
        """Train one epoch."""
        model.train()
        epoch_losses = []
        horizon_losses = {f'horizon_{h}d': [] for h in self.model_config['prediction_horizons']}

        for date in dates:
            # Create batch for date
            batch = dataset.create_graph_batch(
                date,
                lookback_days=self.model_config['historical_window']
            )

            if batch is None:
                continue

            # Move to device
            batch = batch.to(self.device)

            # Forward pass
            predictions = model(
                batch.x,
                batch.edge_index,
                batch.temporal_sequences,
                batch.batch if hasattr(batch, 'batch') else None
            )

            # Calculate loss
            total_loss, individual_losses = model.compute_loss(
                predictions,
                batch.y
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Record losses
            epoch_losses.append(total_loss.item())
            for horizon, loss in individual_losses.items():
                horizon_losses[horizon].append(loss)

        # Average losses
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_horizon_losses = {
            horizon: np.mean(losses) if losses else 0
            for horizon, losses in horizon_losses.items()
        }

        return {
            'total_loss': avg_loss,
            **avg_horizon_losses
        }

    def evaluate(
            self,
            model: TemporalGNN,
            dataset: StockGraphDataset,
            dates: List[pd.Timestamp]
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()

        all_predictions = {h: [] for h in self.model_config['prediction_horizons']}
        all_targets = {h: [] for h in self.model_config['prediction_horizons']}
        all_confidences = {h: [] for h in self.model_config['prediction_horizons']}

        with torch.no_grad():
            for date in dates:
                batch = dataset.create_graph_batch(
                    date,
                    lookback_days=self.model_config['historical_window']
                )

                if batch is None:
                    continue

                batch = batch.to(self.device)

                # Forward pass
                predictions = model(
                    batch.x,
                    batch.edge_index,
                    batch.temporal_sequences,
                    batch.batch if hasattr(batch, 'batch') else None
                )

                # Collect predictions
                for horizon in self.model_config['prediction_horizons']:
                    horizon_key = f'horizon_{horizon}d'
                    pred, conf = predictions[horizon_key]

                    all_predictions[horizon].append(pred.cpu().numpy())
                    all_targets[horizon].append(batch.y[horizon_key].cpu().numpy())
                    all_confidences[horizon].append(conf.cpu().numpy())

        # Calculate metrics
        metrics = {}
        for horizon in self.model_config['prediction_horizons']:
            if all_predictions[horizon]:
                preds = np.concatenate(all_predictions[horizon])
                targets = np.concatenate(all_targets[horizon])
                confs = np.concatenate(all_confidences[horizon])

                horizon_metrics = self.metrics_calculator.calculate_metrics(
                    preds, targets, confs
                )

                for metric_name, value in horizon_metrics.items():
                    metrics[f'{horizon}d_{metric_name}'] = value

        return metrics

    def train(
            self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            graph_structure: Dict,
            num_epochs: int = None,
            learning_rate: float = None,
            batch_size: int = None
    ) -> Dict:
        """Main training loop with Vertex AI tracking."""
        num_epochs = num_epochs or self.model_config['num_epochs']
        learning_rate = learning_rate or self.model_config['learning_rate']
        batch_size = batch_size or self.model_config['batch_size']

        # Start Vertex AI run if enabled
        vertex_run = None
        if self.use_vertex_ai:
            run_id = f"gnn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            vertex_run = aiplatform.start_run(run=run_id, resume=False)

            # Log parameters
            vertex_run.log_params({
                'model_type': 'temporal_gnn',
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'historical_window': self.model_config['historical_window'],
                'hidden_dim': self.gnn_config['hidden_dim'],
                'num_gnn_layers': self.gnn_config['num_gnn_layers'],
                'num_temporal_layers': self.gnn_config['num_temporal_layers'],
                'dropout_rate': self.gnn_config['dropout_rate'],
                'device': str(self.device),
                'num_train_samples': len(train_df),
                'num_val_samples': len(val_df),
                'num_features': len(self.feature_columns),
                'prediction_horizons': str(self.model_config['prediction_horizons'])
            })

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters to MLflow too
            mlflow.log_params({
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'historical_window': self.model_config['historical_window'],
                **self.gnn_config
            })

            # Create datasets
            train_dataset = StockGraphDataset(
                train_df,
                graph_structure,
                sequence_length=self.model_config['historical_window'],
                prediction_horizons=self.model_config['prediction_horizons']
            )

            val_dataset = StockGraphDataset(
                val_df,
                graph_structure,
                sequence_length=self.model_config['historical_window'],
                prediction_horizons=self.model_config['prediction_horizons']
            )

            # Initialize model
            num_features = len(self.feature_columns)
            model = self.initialize_model(num_features)

            # Optimizer and scheduler
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )

            # Training dates
            train_dates = train_df['date'].unique()[-100:]  # Last 100 days for efficiency
            val_dates = val_df['date'].unique()

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                # Train
                train_losses = self.train_epoch(
                    model, train_dataset, optimizer, train_dates
                )

                # Validate
                val_metrics = self.evaluate(
                    model, val_dataset, val_dates
                )

                # Calculate average validation loss
                val_loss = np.mean([
                    val_metrics.get(f'{h}d_mse', 0)
                    for h in self.model_config['prediction_horizons']
                ])

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    self.save_model(model, 'best_model.pth')
                else:
                    patience_counter += 1

                # Prepare metrics for logging
                metrics_dict = {
                    'epoch': epoch,
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in train_losses.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }

                # Log to MLflow
                mlflow.log_metrics(metrics_dict, step=epoch)

                # Log to Vertex AI
                if vertex_run:
                    vertex_run.log_metrics(metrics_dict)

                    # Log time series for Tensorboard
                    vertex_run.log_time_series_metrics({
                        'loss/train': train_losses['total_loss'],
                        'loss/validation': val_loss,
                        'metrics/direction_accuracy': val_metrics.get('1d_direction_accuracy', 0),
                        'metrics/sharpe_ratio': val_metrics.get('7d_sharpe_ratio', 0),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })

                # Log progress
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs} - "
                        f"Train Loss: {train_losses['total_loss']:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )

                # Early stopping
                if patience_counter >= self.model_config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Load best model
            model.load_state_dict(torch.load('best_model.pth'))

            # Final evaluation
            final_metrics = self.evaluate(model, val_dataset, val_dates)

            # Log final metrics to MLflow
            for metric, value in final_metrics.items():
                mlflow.log_metric(f'final_{metric}', value)

            # Log to Vertex AI
            if vertex_run:
                # Log final metrics
                vertex_run.log_metrics({
                    f'final_{metric}': value
                    for metric, value in final_metrics.items()
                })

                # Log model artifact
                model_path = self._save_model_to_gcs(model)

                vertex_run.log_model(
                    model,
                    artifact_id=f"temporal-gnn-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    uri=model_path,
                    metadata={
                        'framework': 'pytorch',
                        'framework_version': torch.__version__,
                        'model_class': 'TemporalGNN',
                        'input_features': len(self.feature_columns),
                        'prediction_horizons': self.model_config['prediction_horizons'],
                        'best_val_loss': float(best_val_loss),
                        'final_metrics': final_metrics,
                        'feature_columns': self.feature_columns,
                        'graph_nodes': len(graph_structure),
                        'graph_edges': sum(len(edges) for edges in graph_structure.values())
                    }
                )

                # End run
                vertex_run.end_run()

            # Save model artifacts
            self.save_training_artifacts(model)

            return {
                'model': model,
                'final_metrics': final_metrics,
                'best_val_loss': best_val_loss,
                'vertex_ai_run': vertex_run.resource_name if vertex_run else None
            }

    def _save_model_to_gcs(self, model: TemporalGNN) -> str:
        """Save model to GCS and return path."""
        from google.cloud import storage

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gcs_path = f"gs://{self.model_config['model_registry_bucket']}/models/temporal_gnn_{timestamp}.pth"

        # Save locally first
        local_path = f"/tmp/model_{timestamp}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self.model_config,
            'gnn_config': self.gnn_config,
            'feature_columns': self.feature_columns
        }, local_path)

        # Upload to GCS
        storage_client = storage.Client()
        bucket_name = self.model_config['model_registry_bucket']
        blob_name = f"models/temporal_gnn_{timestamp}.pth"

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)

        return gcs_path

    def save_model(self, model: TemporalGNN, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_node_features': model.num_node_features,
                'num_edge_features': model.num_edge_features,
                'hidden_dim': model.hidden_dim,
                'prediction_horizons': model.prediction_horizons
            },
            'feature_columns': self.feature_columns,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, filename)
        logger.info(f"Saved model to {filename}")

    def save_training_artifacts(self, model: TemporalGNN):
        """Save all training artifacts."""
        artifacts_dir = Path('training_artifacts')
        artifacts_dir.mkdir(exist_ok=True)

        # Save model
        self.save_model(model, artifacts_dir / 'final_model.pth')

        # Save feature scaler
        import joblib
        joblib.dump(
            self.feature_scaler,
            artifacts_dir / 'feature_scaler.pkl'
        )

        # Save configurations
        config = {
            'model_config': self.model_config,
            'gnn_config': self.gnn_config,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat()
        }

        with open(artifacts_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifacts(str(artifacts_dir))

        logger.info(f"Saved training artifacts to {artifacts_dir}")

    def run_full_training_pipeline(
            self,
            start_date: str,
            end_date: str
    ) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting full training pipeline")

        # Log to Vertex AI experiment
        if self.use_vertex_ai:
            run_params = {
                'pipeline_type': 'full_training',
                'start_date': start_date,
                'end_date': end_date,
                'data_version': f"{start_date}_to_{end_date}"
            }

            if hasattr(self, 'vertex_experiment_run'):
                self.vertex_experiment_run.log_params(run_params)

        # Prepare data
        features_df, graph_structure = self.prepare_training_data(
            start_date, end_date
        )

        # Train/val/test split
        train_df, val_df, test_df = self.create_train_val_test_split(
            features_df
        )

        # Train model
        training_results = self.train(
            train_df, val_df, graph_structure
        )

        # Test evaluation
        test_dataset = StockGraphDataset(
            test_df,
            graph_structure,
            sequence_length=self.model_config['historical_window'],
            prediction_horizons=self.model_config['prediction_horizons']
        )

        test_dates = test_df['date'].unique()
        test_metrics = self.evaluate(
            training_results['model'],
            test_dataset,
            test_dates
        )

        logger.info(f"Test metrics: {test_metrics}")

        # Store model metadata to BigQuery
        self._store_model_metadata(training_results, test_metrics)

        return {
            'model': training_results['model'],
            'val_metrics': training_results['final_metrics'],
            'test_metrics': test_metrics,
            'vertex_ai_run': training_results.get('vertex_ai_run')
        }

    def _store_model_metadata(
            self,
            training_results: Dict,
            test_metrics: Dict
    ):
        """Store model metadata to BigQuery."""
        model_version = f"tgnn_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        metadata = {
            'model_version': model_version,
            'model_type': 'temporal_gnn',
            'training_start_date': self.model_config.get('training_start_date'),
            'training_end_date': self.model_config.get('training_end_date'),
            'validation_metrics': json.dumps(training_results['final_metrics']),
            'test_metrics': json.dumps(test_metrics),
            'hyperparameters': json.dumps({
                **self.model_config,
                **self.gnn_config
            }),
            'model_path': f'gs://{self.model_config["model_registry_bucket"]}/models/{model_version}',
            'vertex_ai_run': training_results.get('vertex_ai_run'),
            'created_at': datetime.now()
        }

        metadata_df = pd.DataFrame([metadata])
        self.bq_client.insert_dataframe(
            metadata_df,
            'model_metadata'
        )