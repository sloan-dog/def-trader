"""
Tests for GNN model and training components.
"""
import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.models.temporal_gnn import TemporalGNN, StockGraphDataset, TemporalAttention
from src.training.trainer import GNNTrainer
from src.training.metrics import ModelMetrics
from src.feature_engineering.graph_constructor import StockGraphConstructor


class TestTemporalGNN:
    """Test Temporal GNN model."""

    @pytest.fixture
    def model_config(self):
        """Model configuration for testing."""
        return {
            'num_node_features': 64,
            'num_edge_features': 0,
            'hidden_dim': 32,
            'num_gnn_layers': 2,
            'num_temporal_layers': 1,
            'num_heads': 2,
            'dropout': 0.1,
            'prediction_horizons': [1, 7, 30, 60]
        }

    @pytest.fixture
    def model(self, model_config):
        """Create test model."""
        return TemporalGNN(**model_config)

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch data."""
        num_nodes = 10
        num_edges = 20
        seq_len = 30
        num_features = 64
        batch_size = 2

        # Node features
        x = torch.randn(num_nodes, num_features)

        # Edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Temporal sequences
        temporal_sequences = torch.randn(batch_size, seq_len, num_features)

        # Batch assignment
        batch = torch.tensor([0] * 5 + [1] * 5)

        # Targets
        targets = {
            f'horizon_{h}d': torch.randn(batch_size)
            for h in [1, 7, 30, 60]
        }

        return {
            'x': x,
            'edge_index': edge_index,
            'temporal_sequences': temporal_sequences,
            'batch': batch,
            'targets': targets
        }

    def test_model_forward(self, model, sample_batch):
        """Test model forward pass."""
        predictions = model(
            sample_batch['x'],
            sample_batch['edge_index'],
            sample_batch['temporal_sequences'],
            sample_batch['batch']
        )

        assert isinstance(predictions, dict)
        assert len(predictions) == 4  # 4 horizons

        for horizon in [1, 7, 30, 60]:
            horizon_key = f'horizon_{horizon}d'
            assert horizon_key in predictions

            pred, conf = predictions[horizon_key]
            assert pred.shape == (2, 1)  # batch_size x 1
            assert conf.shape == (2, 1)  # batch_size x 1
            assert torch.all(conf >= 0) and torch.all(conf <= 1)  # confidence in [0, 1]

    def test_compute_loss(self, model, sample_batch):
        """Test loss computation."""
        predictions = model(
            sample_batch['x'],
            sample_batch['edge_index'],
            sample_batch['temporal_sequences'],
            sample_batch['batch']
        )

        total_loss, individual_losses = model.compute_loss(
            predictions,
            sample_batch['targets']
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert len(individual_losses) == 4

        for horizon_key, loss in individual_losses.items():
            assert isinstance(loss, float)
            assert loss >= 0

    def test_temporal_attention(self):
        """Test temporal attention mechanism."""
        batch_size = 4
        seq_len = 20
        hidden_dim = 32

        attention = TemporalAttention(hidden_dim, num_heads=4)
        x = torch.randn(batch_size, seq_len, hidden_dim)

        output = attention(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestStockGraphDataset:
    """Test dataset creation and handling."""

    @pytest.fixture
    def sample_features_df(self):
        """Create sample features DataFrame."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        tickers = ['AAPL', 'GOOGL', 'MSFT']

        data = []
        for ticker in tickers:
            for date in dates:
                data.append({
                    'ticker': ticker,
                    'date': date,
                    'close': np.random.uniform(100, 200),
                    'volume': np.random.randint(1000000, 10000000),
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.uniform(-2, 2),
                    'sentiment_score': np.random.uniform(-1, 1)
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_graph_structure(self):
        """Create sample graph structure."""
        return {
            'AAPL': {'GOOGL': 0.7, 'MSFT': 0.8},
            'GOOGL': {'AAPL': 0.7, 'MSFT': 0.6},
            'MSFT': {'AAPL': 0.8, 'GOOGL': 0.6}
        }

    def test_dataset_creation(self, sample_features_df, sample_graph_structure):
        """Test dataset creation."""
        dataset = StockGraphDataset(
            sample_features_df,
            sample_graph_structure,
            sequence_length=30,
            prediction_horizons=[1, 7, 30, 60]
        )

        assert len(dataset.tickers) == 3
        assert dataset.sequence_length == 30
        assert dataset.edge_index.shape[0] == 2  # source and target

    def test_create_graph_batch(self, sample_features_df, sample_graph_structure):
        """Test graph batch creation."""
        dataset = StockGraphDataset(
            sample_features_df,
            sample_graph_structure,
            sequence_length=30,
            prediction_horizons=[1, 7]
        )

        target_date = pd.Timestamp('2024-03-01')
        batch = dataset.create_graph_batch(target_date, lookback_days=30)

        assert batch is not None
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'temporal_sequences')
        assert hasattr(batch, 'y')

        # Check shapes
        assert batch.x.shape[0] == 3  # 3 tickers
        assert batch.temporal_sequences.shape[0] == 3  # 3 tickers


class TestModelMetrics:
    """Test metrics calculation."""

    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator."""
        return ModelMetrics()

    @pytest.fixture
    def sample_predictions_actuals(self):
        """Create sample predictions and actuals."""
        n_samples = 100
        predictions = np.random.uniform(-0.05, 0.05, n_samples)
        actuals = predictions + np.random.normal(0, 0.01, n_samples)
        confidences = np.random.uniform(0.5, 0.9, n_samples)

        return predictions, actuals, confidences

    def test_calculate_metrics(self, metrics_calculator, sample_predictions_actuals):
        """Test metrics calculation."""
        predictions, actuals, confidences = sample_predictions_actuals

        metrics = metrics_calculator.calculate_metrics(predictions, actuals, confidences)

        # Check all metrics are present
        expected_metrics = [
            'mse', 'rmse', 'mae', 'r2', 'direction_accuracy',
            'up_precision', 'down_precision', 'return_correlation',
            'return_ic', 'return_rank_ic', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'max_drawdown',
            'var_95', 'cvar_95'
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_directional_accuracy(self, metrics_calculator):
        """Test directional accuracy calculation."""
        predictions = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        actuals = np.array([0.02, -0.01, -0.01, 0.01, 0.03])

        accuracy = metrics_calculator._directional_accuracy(predictions, actuals)

        # 3 out of 5 have same sign
        assert accuracy == 0.6

    def test_sharpe_ratio_calculation(self, metrics_calculator):
        """Test Sharpe ratio calculation."""
        predictions = np.array([1, -1, 1, -1, 1])
        actuals = np.array([0.01, -0.01, 0.02, 0.01, 0.01])

        sharpe = metrics_calculator._calculate_sharpe_ratio(predictions, actuals)

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)


class TestGNNTrainer:
    """Test training pipeline."""

    @pytest.fixture
    def trainer(self):
        """Create test trainer."""
        with patch('src.training.trainer.BigQueryClient'):
            with patch('src.training.trainer.StockGraphConstructor'):
                return GNNTrainer(experiment_name="test")

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')

        data = []
        for date in dates:
            for ticker in ['AAPL', 'GOOGL']:
                data.append({
                    'ticker': ticker,
                    'date': date,
                    'close': np.random.uniform(100, 200),
                    'volume': np.random.randint(1000000, 10000000),
                    'feature1': np.random.randn(),
                    'feature2': np.random.randn()
                })

        return pd.DataFrame(data)

    def test_prepare_training_data(self, trainer, sample_train_data):
        """Test training data preparation."""
        trainer.bq_client.query = Mock(return_value=sample_train_data)
        trainer.graph_constructor.build_graph = Mock(return_value={
            'AAPL': {'GOOGL': 0.8},
            'GOOGL': {'AAPL': 0.8}
        })

        features_df, graph_structure = trainer.prepare_training_data(
            '2023-01-01',
            '2023-04-10'
        )

        assert not features_df.empty
        assert isinstance(graph_structure, dict)
        assert hasattr(trainer, 'feature_scaler')

    def test_train_val_test_split(self, trainer, sample_train_data):
        """Test data splitting."""
        train_df, val_df, test_df = trainer.create_train_val_test_split(
            sample_train_data,
            test_size=0.2,
            val_size=0.2
        )

        total_size = len(sample_train_data)

        # Check sizes (allowing for rounding)
        assert abs(len(train_df) / total_size - 0.6) < 0.1
        assert abs(len(val_df) / total_size - 0.2) < 0.1
        assert abs(len(test_df) / total_size - 0.2) < 0.1

        # Check no overlap
        train_dates = set(train_df['date'])
        val_dates = set(val_df['date'])
        test_dates = set(test_df['date'])

        assert len(train_dates.intersection(val_dates)) == 0
        assert len(val_dates.intersection(test_dates)) == 0


class TestGraphConstructor:
    """Test graph construction."""

    @pytest.fixture
    def constructor(self):
        """Create test constructor."""
        with patch('src.feature_engineering.graph_constructor.BigQueryClient'):
            return StockGraphConstructor()

    def test_calculate_edge_weight(self, constructor):
        """Test edge weight calculation."""
        metadata = {
            'AAPL': {'sector': 'technology', 'market_cap': 'large'},
            'MSFT': {'sector': 'technology', 'market_cap': 'large'},
            'JPM': {'sector': 'financials', 'market_cap': 'large'}
        }

        correlations = pd.DataFrame({
            'AAPL': {'AAPL': 1.0, 'MSFT': 0.8, 'JPM': 0.3},
            'MSFT': {'AAPL': 0.8, 'MSFT': 1.0, 'JPM': 0.4},
            'JPM': {'AAPL': 0.3, 'JPM': 0.4, 'JPM': 1.0}
        })

        # Same sector, high correlation
        weight1 = constructor._calculate_edge_weight('AAPL', 'MSFT', metadata, correlations)

        # Different sector, low correlation
        weight2 = constructor._calculate_edge_weight('AAPL', 'JPM', metadata, correlations)

        assert weight1 > weight2
        assert 0 <= weight1 <= 1
        assert 0 <= weight2 <= 1


# Integration tests
class TestModelIntegration:
    """Integration tests for model pipeline."""

    @pytest.mark.integration
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # This would test the full training flow
        # Requires actual data and compute resources
        pass

    @pytest.mark.integration
    def test_prediction_pipeline(self):
        """Test prediction generation pipeline."""
        # This would test prediction generation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])