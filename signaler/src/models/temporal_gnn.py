"""
Temporal Graph Neural Network for multi-horizon stock prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict, Optional
import numpy as np
from src.utils import logger
from config.settings import GNN_CONFIG, MODEL_CONFIG


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask

        Returns:
            Output tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for stock relationships."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int = 4,
            dropout: float = 0.2
    ):
        super().__init__()
        self.gat_conv = GATConv(
            in_features,
            out_features // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply graph attention."""
        x = self.gat_conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class TemporalGraphBlock(nn.Module):
    """Combined temporal and graph processing block."""

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 4,
            dropout: float = 0.2
    ):
        super().__init__()

        # Graph processing
        self.graph_attention = GraphAttentionLayer(
            hidden_dim, hidden_dim, num_heads, dropout
        )

        # Temporal processing
        self.temporal_attention = TemporalAttention(hidden_dim, num_heads)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout
        )

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            temporal_features: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process both graph and temporal information.

        Args:
            node_features: Node features (num_nodes, hidden_dim)
            temporal_features: Temporal sequences (batch_size, seq_len, hidden_dim)
            edge_index: Graph edge indices
            batch: Batch assignment for nodes

        Returns:
            Updated node and temporal features
        """
        # Graph processing
        graph_out = self.graph_attention(node_features, edge_index)

        # Temporal processing
        temporal_attn = self.temporal_attention(temporal_features)
        temporal_lstm, _ = self.lstm(temporal_features)

        # Aggregate graph features for each sequence
        if batch is not None:
            # Pool graph features by batch
            pooled_graph = global_mean_pool(graph_out, batch)
            pooled_graph = pooled_graph.unsqueeze(1).expand(-1, temporal_lstm.size(1), -1)
        else:
            pooled_graph = graph_out.mean(0, keepdim=True).expand(
                temporal_lstm.size(0), temporal_lstm.size(1), -1
            )

        # Fuse graph and temporal features
        fused_features = torch.cat([temporal_lstm, pooled_graph], dim=-1)
        fused_features = self.fusion_layer(fused_features)

        return graph_out, fused_features


class TemporalGNN(nn.Module):
    """
    Temporal Graph Neural Network for multi-horizon stock prediction.

    This model combines:
    1. Graph neural networks to capture cross-stock relationships
    2. Temporal attention and LSTM for time series patterns
    3. Multi-horizon prediction heads
    """

    def __init__(
            self,
            num_node_features: int,
            num_edge_features: int,
            hidden_dim: int = 128,
            num_gnn_layers: int = 3,
            num_temporal_layers: int = 2,
            num_heads: int = 4,
            dropout: float = 0.2,
            prediction_horizons: List[int] = [1, 7, 30, 60]
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.prediction_horizons = prediction_horizons

        # Input projections
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.edge_encoder = nn.Linear(num_edge_features, hidden_dim) if num_edge_features > 0 else None

        # Temporal-Graph blocks
        self.tg_blocks = nn.ModuleList([
            TemporalGraphBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # Additional temporal processing
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_temporal_layers
        )

        # Prediction heads for different horizons
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}d': self._create_prediction_head(hidden_dim, dropout)
            for h in prediction_horizons
        })

        # Confidence estimation heads
        self.confidence_heads = nn.ModuleDict({
            f'horizon_{h}d': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for h in prediction_horizons
        })

    def _create_prediction_head(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Create a prediction head for a specific horizon."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            temporal_sequences: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Temporal GNN.

        Args:
            node_features: Node feature matrix (num_nodes, num_features)
            edge_index: Graph connectivity (2, num_edges)
            temporal_sequences: Time series data (batch_size, seq_len, num_features)
            batch: Batch vector for node assignment
            edge_attr: Edge features (optional)

        Returns:
            Dictionary with predictions and confidences for each horizon
        """
        # Encode inputs
        node_embeddings = self.node_encoder(node_features)

        # Initial temporal features (use node features aggregated over time)
        temporal_features = temporal_sequences

        # Process through temporal-graph blocks
        for block in self.tg_blocks:
            node_embeddings, temporal_features = block(
                node_embeddings, temporal_features, edge_index, batch
            )

        # Additional temporal processing
        temporal_output = self.temporal_transformer(temporal_features)

        # Get final representation (last timestamp)
        final_representation = temporal_output[:, -1, :]

        # Generate predictions for each horizon
        predictions = {}
        for horizon in self.prediction_horizons:
            horizon_key = f'horizon_{horizon}d'

            # Prediction
            pred = self.prediction_heads[horizon_key](final_representation)

            # Confidence
            conf = self.confidence_heads[horizon_key](final_representation)

            predictions[horizon_key] = (pred, conf)

        return predictions

    def compute_loss(
            self,
            predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            targets: Dict[str, torch.Tensor],
            loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-horizon prediction loss.

        Args:
            predictions: Model predictions with confidences
            targets: Ground truth values for each horizon
            loss_weights: Optional weights for each horizon

        Returns:
            Total loss and individual losses per horizon
        """
        if loss_weights is None:
            loss_weights = {h: 1.0 for h in predictions.keys()}

        total_loss = 0
        individual_losses = {}

        for horizon_key in predictions.keys():
            pred, conf = predictions[horizon_key]
            target = targets[horizon_key]

            # Prediction loss (weighted MSE)
            pred_loss = F.mse_loss(pred.squeeze(), target)

            # Confidence-weighted loss
            weighted_loss = (conf.squeeze() * pred_loss).mean()

            # Confidence calibration loss
            # Encourage confidence to be high when prediction error is low
            pred_error = torch.abs(pred.squeeze() - target)
            normalized_error = pred_error / (pred_error.mean() + 1e-8)
            conf_loss = F.binary_cross_entropy(
                conf.squeeze(),
                1 - torch.sigmoid(normalized_error)
            )

            # Combined loss for this horizon
            horizon_loss = weighted_loss + 0.1 * conf_loss

            total_loss += loss_weights[horizon_key] * horizon_loss
            individual_losses[horizon_key] = horizon_loss.item()

        return total_loss, individual_losses


class StockGraphDataset:
    """Dataset handler for temporal graph data."""

    def __init__(
            self,
            features_df: pd.DataFrame,
            graph_structure: Dict,
            sequence_length: int = 90,
            prediction_horizons: List[int] = [1, 7, 30, 60]
    ):
        """
        Initialize dataset.

        Args:
            features_df: DataFrame with all features
            graph_structure: Dictionary defining graph edges
            sequence_length: Length of input sequences
            prediction_horizons: List of prediction horizons
        """
        self.features_df = features_df
        self.graph_structure = graph_structure
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for training."""
        # Group by ticker
        self.ticker_groups = self.features_df.groupby('ticker')
        self.tickers = list(self.ticker_groups.groups.keys())

        # Create ticker to index mapping
        self.ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.tickers)}

        # Build edge index
        self._build_edge_index()

        # Identify feature columns
        exclude_cols = ['ticker', 'date', 'inserted_at']
        self.feature_cols = [
            col for col in self.features_df.columns
            if col not in exclude_cols
        ]

    def _build_edge_index(self):
        """Build edge index from graph structure."""
        edges = []

        for source_ticker, connections in self.graph_structure.items():
            if source_ticker not in self.ticker_to_idx:
                continue

            source_idx = self.ticker_to_idx[source_ticker]

            for target_ticker, weight in connections.items():
                if target_ticker not in self.ticker_to_idx:
                    continue

                target_idx = self.ticker_to_idx[target_ticker]
                edges.append([source_idx, target_idx])

        self.edge_index = torch.tensor(edges, dtype=torch.long).t()

    def create_graph_batch(
            self,
            date: pd.Timestamp,
            lookback_days: int = 90
    ) -> Optional[Data]:
        """
        Create a graph batch for a specific date.

        Args:
            date: Target date
            lookback_days: Number of days to look back

        Returns:
            PyTorch Geometric Data object or None if insufficient data
        """
        start_date = date - pd.Timedelta(days=lookback_days)

        # Collect node features for all tickers
        node_features_list = []
        temporal_sequences_list = []
        valid_tickers = []

        for ticker in self.tickers:
            ticker_data = self.ticker_groups.get_group(ticker)

            # Filter date range
            mask = (ticker_data['date'] > start_date) & (ticker_data['date'] <= date)
            sequence_data = ticker_data[mask].sort_values('date')

            if len(sequence_data) < lookback_days * 0.8:  # Require at least 80% data
                continue

            # Extract features
            features = sequence_data[self.feature_cols].values

            # Get latest features for node representation
            node_features_list.append(features[-1])

            # Get full sequence for temporal processing
            temporal_sequences_list.append(features)

            valid_tickers.append(ticker)

        if len(valid_tickers) < 10:  # Require minimum number of stocks
            return None

        # Create tensors
        node_features = torch.tensor(
            np.array(node_features_list),
            dtype=torch.float32
        )

        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in temporal_sequences_list)
        padded_sequences = []

        for seq in temporal_sequences_list:
            if seq.shape[0] < max_len:
                padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
                seq = np.vstack([padding, seq])
            padded_sequences.append(seq)

        temporal_sequences = torch.tensor(
            np.array(padded_sequences),
            dtype=torch.float32
        )

        # Filter edge index for valid tickers
        valid_indices = [self.ticker_to_idx[t] for t in valid_tickers]
        valid_mask = torch.tensor([
            (self.edge_index[0, i] in valid_indices) and
            (self.edge_index[1, i] in valid_indices)
            for i in range(self.edge_index.shape[1])
        ])

        filtered_edge_index = self.edge_index[:, valid_mask]

        # Create targets for different horizons
        targets = {}
        for horizon in self.prediction_horizons:
            horizon_targets = []

            for ticker in valid_tickers:
                ticker_data = self.ticker_groups.get_group(ticker)

                # Get future price
                future_date = date + pd.Timedelta(days=horizon)
                future_data = ticker_data[ticker_data['date'] == future_date]

                if not future_data.empty:
                    current_price = ticker_data[ticker_data['date'] == date]['close'].iloc[0]
                    future_price = future_data['close'].iloc[0]
                    return_val = (future_price - current_price) / current_price
                    horizon_targets.append(return_val)
                else:
                    horizon_targets.append(0.0)  # Will be masked in loss

            targets[f'horizon_{horizon}d'] = torch.tensor(
                horizon_targets,
                dtype=torch.float32
            )

        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=filtered_edge_index,
            temporal_sequences=temporal_sequences,
            y=targets,
            num_nodes=len(valid_tickers)
        )

        return data