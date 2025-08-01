"""
Model evaluation metrics for trading signals.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from src.utils import logger
class ModelMetrics:
    """Calculate comprehensive metrics for model evaluation."""

    def calculate_metrics(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            confidences: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all metrics for predictions."""
        metrics = {}

        # Basic regression metrics
        metrics['mse'] = mean_squared_error(actuals, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(actuals, predictions)
        metrics['r2'] = r2_score(actuals, predictions)

        # Directional accuracy
        metrics['direction_accuracy'] = self._directional_accuracy(predictions, actuals)
        metrics['up_precision'] = self._directional_precision(predictions, actuals, direction='up')
        metrics['down_precision'] = self._directional_precision(predictions, actuals, direction='down')

        # Return distribution metrics
        metrics['return_correlation'] = np.corrcoef(predictions, actuals)[0, 1]
        metrics['return_ic'] = self._information_coefficient(predictions, actuals)
        metrics['return_rank_ic'] = self._rank_information_coefficient(predictions, actuals)

        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(predictions, actuals)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(predictions, actuals)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(predictions, actuals)

        # Tail risk metrics
        metrics['max_drawdown'] = self._calculate_max_drawdown(predictions, actuals)
        metrics['var_95'] = self._calculate_var(predictions, actuals, 0.95)
        metrics['cvar_95'] = self._calculate_cvar(predictions, actuals, 0.95)

        # Confidence calibration (if provided)
        if confidences is not None:
            conf_metrics = self._confidence_metrics(predictions, actuals, confidences)
            metrics.update(conf_metrics)

        # Trading metrics
        trading_metrics = self._calculate_trading_metrics(predictions, actuals)
        metrics.update(trading_metrics)

        return metrics

    def _directional_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate percentage of correct direction predictions."""
        correct_direction = (np.sign(predictions) == np.sign(actuals))
        return np.mean(correct_direction)

    def _directional_precision(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            direction: str = 'up'
    ) -> float:
        """Calculate precision for specific direction."""
        if direction == 'up':
            predicted_mask = predictions > 0
            actual_positive = actuals > 0
        else:
            predicted_mask = predictions < 0
            actual_positive = actuals < 0

        if np.sum(predicted_mask) == 0:
            return 0.0

        true_positives = np.sum(predicted_mask & actual_positive)
        precision = true_positives / np.sum(predicted_mask)

        return precision

    def _information_coefficient(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate information coefficient (IC)."""
        # Remove any NaN values
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if np.sum(mask) < 10:
            return 0.0

        return np.corrcoef(predictions[mask], actuals[mask])[0, 1]

    def _rank_information_coefficient(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray
    ) -> float:
        """Calculate rank IC using Spearman correlation."""
        mask = ~(np.isnan(predictions) | np.isnan(actuals))
        if np.sum(mask) < 10:
            return 0.0

        return stats.spearmanr(predictions[mask], actuals[mask])[0]

    def _calculate_sharpe_ratio(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio of prediction-based strategy."""
        # Simple strategy: long if prediction > 0, short if < 0
        strategy_returns = np.sign(predictions) * actuals

        excess_returns = strategy_returns - risk_free_rate / 252  # Daily risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def _calculate_sortino_ratio(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        strategy_returns = np.sign(predictions) * actuals
        excess_returns = strategy_returns - risk_free_rate / 252

        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return sortino

    def _calculate_calmar_ratio(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        strategy_returns = np.sign(predictions) * actuals

        annual_return = np.mean(strategy_returns) * 252
        max_dd = self._calculate_max_drawdown(predictions, actuals)

        if max_dd == 0:
            return 0.0

        return annual_return / abs(max_dd)

    def _calculate_max_drawdown(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray
    ) -> float:
        """Calculate maximum drawdown of strategy."""
        strategy_returns = np.sign(predictions) * actuals
        cumulative_returns = np.cumprod(1 + strategy_returns)

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        return np.min(drawdown)

    def _calculate_var(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        strategy_returns = np.sign(predictions) * actuals
        return np.percentile(strategy_returns, (1 - confidence) * 100)

    def _calculate_cvar(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        strategy_returns = np.sign(predictions) * actuals
        var = self._calculate_var(predictions, actuals, confidence)

        tail_returns = strategy_returns[strategy_returns <= var]

        if len(tail_returns) == 0:
            return var

        return np.mean(tail_returns)

    def _confidence_metrics(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            confidences: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        metrics = {}

        # Bin predictions by confidence
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)

        calibration_error = 0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_confidence = np.mean(confidences[mask])
                bin_accuracy = self._directional_accuracy(
                    predictions[mask],
                    actuals[mask]
                )
                calibration_error += np.abs(bin_confidence - bin_accuracy) * np.sum(mask)

        metrics['expected_calibration_error'] = calibration_error / len(predictions)

        # Confidence-weighted accuracy
        high_conf_mask = confidences > 0.7
        if np.sum(high_conf_mask) > 0:
            metrics['high_confidence_accuracy'] = self._directional_accuracy(
                predictions[high_conf_mask],
                actuals[high_conf_mask]
            )
            metrics['high_confidence_ratio'] = np.mean(high_conf_mask)

        return metrics

    def _calculate_trading_metrics(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate practical trading metrics."""
        metrics = {}

        # Assuming predictions are returns, calculate position sizes
        positions = np.sign(predictions)

        # Win rate
        winning_trades = (positions * actuals) > 0
        metrics['win_rate'] = np.mean(winning_trades)

        # Average win/loss
        wins = actuals[winning_trades]
        losses = actuals[~winning_trades]

        if len(wins) > 0:
            metrics['avg_win'] = np.mean(np.abs(wins))
        else:
            metrics['avg_win'] = 0

        if len(losses) > 0:
            metrics['avg_loss'] = np.mean(np.abs(losses))
        else:
            metrics['avg_loss'] = 0

        # Profit factor
        if metrics['avg_loss'] > 0 and len(wins) > 0 and len(losses) > 0:
            metrics['profit_factor'] = (
                    metrics['avg_win'] * len(wins) /
                    (metrics['avg_loss'] * len(losses))
            )
        else:
            metrics['profit_factor'] = 0

        # Hit rate for different return thresholds
        for threshold in [0.01, 0.02, 0.05]:  # 1%, 2%, 5%
            hit_rate = np.mean(
                (positions > 0) & (actuals > threshold) |
                (positions < 0) & (actuals < -threshold)
            )
            metrics[f'hit_rate_{int(threshold*100)}pct'] = hit_rate

        return metrics

    def calculate_portfolio_metrics(
            self,
            predictions_dict: Dict[str, np.ndarray],
            actuals_dict: Dict[str, np.ndarray],
            weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate metrics for portfolio of predictions."""
        if weights is None:
            weights = {ticker: 1.0 / len(predictions_dict) for ticker in predictions_dict}

        # Aggregate portfolio returns
        portfolio_predictions = []
        portfolio_actuals = []

        for ticker, pred in predictions_dict.items():
            weight = weights.get(ticker, 0)
            portfolio_predictions.append(pred * weight)
            portfolio_actuals.append(actuals_dict[ticker] * weight)

        portfolio_predictions = np.sum(portfolio_predictions, axis=0)
        portfolio_actuals = np.sum(portfolio_actuals, axis=0)

        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_metrics(portfolio_predictions, portfolio_actuals)

        # Add diversification metrics
        correlations = []
        for t1 in predictions_dict:
            for t2 in predictions_dict:
                if t1 < t2:
                    corr = np.corrcoef(predictions_dict[t1], predictions_dict[t2])[0, 1]
                    correlations.append(corr)

        portfolio_metrics['avg_correlation'] = np.mean(correlations) if correlations else 0
        portfolio_metrics['diversification_ratio'] = 1 - portfolio_metrics['avg_correlation']

        return portfolio_metrics

    def create_performance_report(
            self,
            predictions: pd.DataFrame,
            actuals: pd.DataFrame,
            metadata: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Create comprehensive performance report."""
        report_data = []

        # Calculate metrics for each horizon
        for horizon in ['1d', '7d', '30d', '60d']:
            pred_col = f'horizon_{horizon}'
            actual_col = f'actual_{horizon}'

            if pred_col in predictions.columns and actual_col in actuals.columns:
                metrics = self.calculate_metrics(
                    predictions[pred_col].values,
                    actuals[actual_col].values
                )

                metrics['horizon'] = horizon
                if metadata:
                    metrics.update(metadata)

                report_data.append(metrics)

        report_df = pd.DataFrame(report_data)

        # Add summary statistics
        summary_row = {
            'horizon': 'average',
            'direction_accuracy': report_df['direction_accuracy'].mean(),
            'sharpe_ratio': report_df['sharpe_ratio'].mean(),
            'max_drawdown': report_df['max_drawdown'].mean()
        }

        report_df = pd.concat([
            report_df,
            pd.DataFrame([summary_row])
        ], ignore_index=True)

        return report_df