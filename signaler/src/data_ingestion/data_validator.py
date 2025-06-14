"""
Data validation utilities for ensuring data quality.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from loguru import logger


class DataValidator:
    """Validate financial data for quality and consistency."""

    def __init__(self):
        """Initialize validator with validation rules."""
        self.ohlcv_rules = {
            'price_consistency': self._check_price_consistency,
            'volume_validity': self._check_volume_validity,
            'date_continuity': self._check_date_continuity,
            'price_outliers': self._check_price_outliers,
            'data_types': self._check_data_types
        }

        self.indicator_rules = {
            'rsi_range': lambda df: self._check_range(df, 'rsi', 0, 100),
            'correlation_range': lambda df: self._check_range(df, 'correlation', -1, 1),
            'percentage_range': lambda df: self._check_percentage_columns(df),
            'nan_threshold': lambda df: self._check_nan_threshold(df, 0.3)
        }

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate OHLCV data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
            return False, issues

        # Run validation rules
        for rule_name, rule_func in self.ohlcv_rules.items():
            try:
                is_valid, issue = rule_func(df)
                if not is_valid:
                    issues.append(f"{rule_name}: {issue}")
            except Exception as e:
                issues.append(f"{rule_name}: Error during validation - {str(e)}")

        return len(issues) == 0, issues

    def validate_indicators(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate technical indicators data."""
        issues = []

        # Run indicator validation rules
        for rule_name, rule_func in self.indicator_rules.items():
            try:
                is_valid, issue = rule_func(df)
                if not is_valid:
                    issues.append(f"{rule_name}: {issue}")
            except Exception as e:
                issues.append(f"{rule_name}: Error during validation - {str(e)}")

        return len(issues) == 0, issues

    def validate_predictions(
            self,
            predictions: pd.DataFrame,
            actuals: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate model predictions against actuals."""
        metrics = {}

        # Check prediction range
        for col in predictions.columns:
            if 'horizon' in col:
                # Predictions should be reasonable returns (-50% to +50%)
                outliers = (predictions[col].abs() > 0.5).sum()
                metrics[f'{col}_outliers'] = outliers

                if outliers > len(predictions) * 0.01:  # More than 1% outliers
                    metrics[f'{col}_warning'] = "High number of extreme predictions"

        # If actuals available, calculate accuracy metrics
        if actuals is not None and not actuals.empty:
            for horizon in ['1d', '7d', '30d', '60d']:
                pred_col = f'horizon_{horizon}'
                actual_col = f'actual_{horizon}'

                if pred_col in predictions.columns and actual_col in actuals.columns:
                    # Calculate metrics
                    mae = (predictions[pred_col] - actuals[actual_col]).abs().mean()
                    rmse = np.sqrt(((predictions[pred_col] - actuals[actual_col]) ** 2).mean())
                    direction_accuracy = (
                            (predictions[pred_col] > 0) == (actuals[actual_col] > 0)
                    ).mean()

                    metrics[f'{horizon}_mae'] = mae
                    metrics[f'{horizon}_rmse'] = rmse
                    metrics[f'{horizon}_direction_accuracy'] = direction_accuracy

        return metrics

    def _check_price_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check OHLC price relationships."""
        invalid_rows = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
            ]

        if len(invalid_rows) > 0:
            return False, f"Found {len(invalid_rows)} rows with invalid OHLC relationships"
        return True, ""

    def _check_volume_validity(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check volume data validity."""
        # Check for negative volumes
        negative_volumes = (df['volume'] < 0).sum()
        if negative_volumes > 0:
            return False, f"Found {negative_volumes} negative volume values"

        # Check for excessive zero volumes
        zero_volumes = (df['volume'] == 0).sum()
        zero_volume_pct = zero_volumes / len(df) * 100

        if zero_volume_pct > 10:  # More than 10% zero volume
            return False, f"Found {zero_volume_pct:.1f}% zero volume days"

        return True, ""

    def _check_date_continuity(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for missing dates in time series."""
        if 'date' not in df.columns:
            return True, ""  # Skip if no date column

        df_sorted = df.sort_values('date')
        date_diffs = pd.to_datetime(df_sorted['date']).diff()

        # Check for large gaps (more than 10 business days)
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=14)]

        if len(large_gaps) > 0:
            return False, f"Found {len(large_gaps)} large gaps in date sequence"

        return True, ""

    def _check_price_outliers(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for price outliers using statistical methods."""
        price_cols = ['open', 'high', 'low', 'close']
        outlier_counts = {}

        for col in price_cols:
            if col in df.columns:
                # Calculate rolling statistics
                rolling_mean = df[col].rolling(20, min_periods=1).mean()
                rolling_std = df[col].rolling(20, min_periods=1).std()

                # Identify outliers (more than 5 std deviations)
                outliers = abs(df[col] - rolling_mean) > (5 * rolling_std)
                outlier_counts[col] = outliers.sum()

        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            return False, f"Found price outliers: {outlier_counts}"

        return True, ""

    def _check_data_types(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check data types are correct."""
        expected_types = {
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64',
            'adjusted_close': 'float64'
        }

        type_issues = []
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                # Allow compatible types
                if expected_type == 'float64' and actual_type not in ['float64', 'float32']:
                    type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
                elif expected_type == 'int64' and actual_type not in ['int64', 'int32', 'float64']:
                    type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")

        if type_issues:
            return False, f"Data type issues: {type_issues}"

        return True, ""

    def _check_range(
            self,
            df: pd.DataFrame,
            column: str,
            min_val: float,
            max_val: float
    ) -> Tuple[bool, str]:
        """Check if column values are within expected range."""
        if column not in df.columns:
            return True, ""

        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()

        if out_of_range > 0:
            return False, f"{column} has {out_of_range} values outside range [{min_val}, {max_val}]"

        return True, ""

    def _check_percentage_columns(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check percentage columns are in valid range."""
        pct_columns = [col for col in df.columns if 'percent' in col.lower() or 'pct' in col.lower()]

        issues = []
        for col in pct_columns:
            # Most percentages should be between -100% and 100%
            extremes = ((df[col] < -1) | (df[col] > 1)).sum()
            if extremes > 0:
                issues.append(f"{col}: {extremes} extreme values")

        if issues:
            return False, f"Percentage column issues: {issues}"

        return True, ""

    def _check_nan_threshold(
            self,
            df: pd.DataFrame,
            threshold: float = 0.3
    ) -> Tuple[bool, str]:
        """Check if NaN values exceed threshold."""
        nan_ratios = df.isna().sum() / len(df)
        high_nan_cols = nan_ratios[nan_ratios > threshold]

        if len(high_nan_cols) > 0:
            return False, f"Columns with >={threshold*100}% NaN: {high_nan_cols.to_dict()}"

        return True, ""

    def generate_validation_report(
            self,
            df: pd.DataFrame,
            data_type: str = 'ohlcv'
    ) -> pd.DataFrame:
        """Generate detailed validation report."""
        report_data = {
            'column': [],
            'non_null_count': [],
            'null_percentage': [],
            'unique_values': [],
            'min': [],
            'max': [],
            'mean': [],
            'std': [],
            'validation_status': []
        }

        for col in df.columns:
            report_data['column'].append(col)
            report_data['non_null_count'].append(df[col].notna().sum())
            report_data['null_percentage'].append(df[col].isna().sum() / len(df) * 100)
            report_data['unique_values'].append(df[col].nunique())

            # Numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                report_data['min'].append(df[col].min())
                report_data['max'].append(df[col].max())
                report_data['mean'].append(df[col].mean())
                report_data['std'].append(df[col].std())
            else:
                report_data['min'].append(None)
                report_data['max'].append(None)
                report_data['mean'].append(None)
                report_data['std'].append(None)

            # Column-specific validation
            col_valid = True
            if data_type == 'ohlcv' and col in ['open', 'high', 'low', 'close']:
                col_valid = (df[col] > 0).all() if col in df.columns else False

            report_data['validation_status'].append('PASS' if col_valid else 'FAIL')

        return pd.DataFrame(report_data)