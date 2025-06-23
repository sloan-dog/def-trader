"""
Technical indicators calculation module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from loguru import logger

from config.settings import TECHNICAL_INDICATORS
from datetime import datetime


class TechnicalIndicatorCalculator:
    """Calculate technical indicators from OHLCV data."""

    def __init__(self, config: Dict = None):
        """Initialize calculator with configuration."""
        self.config = config or TECHNICAL_INDICATORS

    def calculate_all_indicators(
            self,
            df: pd.DataFrame,
            ticker: str = None
    ) -> pd.DataFrame:
        """Calculate all configured technical indicators."""
        # Ensure data is sorted by date
        df = df.sort_values('date').copy()

        # Create a copy for calculations
        calc_df = df.copy()

        # Calculate each indicator
        if 'rsi' in self.config:
            calc_df = self._calculate_rsi(calc_df, **self.config['rsi'])

        if 'ema' in self.config:
            calc_df = self._calculate_ema(calc_df, **self.config['ema'])

        if 'sma' in self.config:
            calc_df = self._calculate_sma(calc_df, **self.config['sma'])

        if 'macd' in self.config:
            calc_df = self._calculate_macd(calc_df, **self.config['macd'])

        if 'bollinger_bands' in self.config:
            calc_df = self._calculate_bollinger_bands(calc_df, **self.config['bollinger_bands'])

        if 'atr' in self.config:
            calc_df = self._calculate_atr(calc_df, **self.config['atr'])

        if 'adx' in self.config:
            calc_df = self._calculate_adx(calc_df, **self.config['adx'])

        if 'obv' in self.config:
            calc_df = self._calculate_obv(calc_df)

        if 'vwap' in self.config:
            calc_df = self._calculate_vwap(calc_df)

        # Add additional calculated features
        calc_df = self._calculate_price_features(calc_df)
        calc_df = self._calculate_volume_features(calc_df)

        if ticker:
            calc_df['ticker'] = ticker

        return calc_df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        df[f'rsi'] = ta.momentum.RSIIndicator(
            close=df['close'],
            window=period
        ).rsi()

        # Add RSI-based features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        return df

    def _calculate_ema(self, df: pd.DataFrame, periods: List[int] = [9, 20, 50]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = ta.trend.EMAIndicator(
                close=df['close'],
                window=period
            ).ema_indicator()

            # Calculate price position relative to EMA
            df[f'close_to_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

        # EMA crossovers
        if 9 in periods and 20 in periods:
            df['ema_9_20_cross'] = (
                    (df['ema_9'] > df['ema_20']) &
                    (df['ema_9'].shift(1) <= df['ema_20'].shift(1))
            ).astype(int)

        return df

    def _calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        for period in periods:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(
                close=df['close'],
                window=period
            ).sma_indicator()

            # Golden/Death cross signals
            if period == 50 and 'sma_200' in df.columns:
                df['golden_cross'] = (
                        (df['sma_50'] > df['sma_200']) &
                        (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
                ).astype(int)

                df['death_cross'] = (
                        (df['sma_50'] < df['sma_200']) &
                        (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
                ).astype(int)

        return df

    def _calculate_macd(
            self,
            df: pd.DataFrame,
            fast: int = 12,
            slow: int = 26,
            signal: int = 9
    ) -> pd.DataFrame:
        """Calculate MACD indicator."""
        macd_indicator = ta.trend.MACD(
            close=df['close'],
            window_slow=slow,
            window_fast=fast,
            window_sign=signal
        )

        df['macd'] = macd_indicator.macd()
        df['macd_signal'] = macd_indicator.macd_signal()
        df['macd_hist'] = macd_indicator.macd_diff()

        # MACD crossovers
        df['macd_cross_above'] = (
                (df['macd'] > df['macd_signal']) &
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)

        df['macd_cross_below'] = (
                (df['macd'] < df['macd_signal']) &
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        ).astype(int)

        return df

    def _calculate_bollinger_bands(
            self,
            df: pd.DataFrame,
            period: int = 20,
            std_dev: int = 2
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        bb_indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=period,
            window_dev=std_dev
        )

        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_width'] = bb_indicator.bollinger_wband()
        df['bb_percent'] = bb_indicator.bollinger_pband()

        # Bollinger Band signals
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        df['bb_overbought'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_oversold'] = (df['close'] < df['bb_lower']).astype(int)

        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        ).average_true_range()

        # ATR percentage (normalized by price)
        df['atr_percent'] = df['atr'] / df['close'] * 100

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        adx_indicator = ta.trend.ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=period
        )

        df['adx'] = adx_indicator.adx()
        df['adx_pos'] = adx_indicator.adx_pos()
        df['adx_neg'] = adx_indicator.adx_neg()

        # Trend strength
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend'] = (df['adx'] < 20).astype(int)

        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()

        # OBV trend
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        df['obv_trend_up'] = (df['obv'] > df['obv_ema']).astype(int)

        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = df['typical_price'] * df['volume']

        # Reset at each day for intraday VWAP
        df['vwap'] = (
                df.groupby(df['date'])['price_volume'].cumsum() /
                df.groupby(df['date'])['volume'].cumsum()
        )

        # VWAP deviation
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100

        # Clean up temporary columns
        df = df.drop(['typical_price', 'price_volume'], axis=1)

        return df

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional price-based features."""
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_20d'] = df['close'].pct_change(20)

        # Highs and lows
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']

        # Rolling highs/lows
        df['rolling_high_20d'] = df['high'].rolling(20).max()
        df['rolling_low_20d'] = df['low'].rolling(20).min()
        df['price_position_20d'] = (
                (df['close'] - df['rolling_low_20d']) /
                (df['rolling_high_20d'] - df['rolling_low_20d'])
        )

        # Volatility
        df['volatility_20d'] = df['price_change'].rolling(20).std()
        df['volatility_60d'] = df['price_change'].rolling(60).std()

        # Price momentum
        df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_30d'] = df['close'] / df['close'].shift(30) - 1

        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_10d'] = df['volume'].rolling(10).mean()
        df['volume_ma_20d'] = df['volume'].rolling(20).mean()

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma_20d']

        # Price-volume correlation
        df['price_volume_corr'] = (
            df['close'].pct_change()
            .rolling(20)
            .corr(df['volume'].pct_change())
        )

        # Volume spikes
        volume_std = df['volume'].rolling(20).std()
        volume_mean = df['volume'].rolling(20).mean()
        df['volume_spike'] = (
            (df['volume'] > volume_mean + 2 * volume_std).astype(int)
        )

        return df

    def validate_indicators(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Validate calculated indicators and return quality metrics."""
        validation_results = {}

        # List of expected indicator columns
        expected_indicators = [
            'rsi', 'ema_9', 'ema_20', 'ema_50', 'sma_20', 'sma_50',
            'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr', 'adx', 'obv', 'vwap'
        ]

        for indicator in expected_indicators:
            if indicator in df.columns:
                validation_results[indicator] = {
                    'exists': True,
                    'null_count': df[indicator].isnull().sum(),
                    'null_percentage': df[indicator].isnull().sum() / len(df) * 100,
                    'min': df[indicator].min(),
                    'max': df[indicator].max(),
                    'mean': df[indicator].mean(),
                    'std': df[indicator].std()
                }
            else:
                validation_results[indicator] = {
                    'exists': False,
                    'error': 'Indicator not calculated'
                }

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count

        if inf_counts:
            validation_results['infinite_values'] = inf_counts

        return validation_results

    def create_indicator_features(
            self,
            df: pd.DataFrame,
            lookback_periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """Create lagged and rolling features from indicators."""
        feature_df = df.copy()

        # Indicators to create features for
        indicators = [
            'rsi', 'macd', 'macd_hist', 'bb_percent', 'atr_percent',
            'adx', 'volume_ratio', 'momentum_10d', 'volatility_20d'
        ]

        for indicator in indicators:
            if indicator not in feature_df.columns:
                continue

            # Lagged features
            for lag in lookback_periods:
                feature_df[f'{indicator}_lag_{lag}'] = feature_df[indicator].shift(lag)

            # Rolling statistics
            for period in [5, 10, 20]:
                feature_df[f'{indicator}_ma_{period}'] = (
                    feature_df[indicator].rolling(period).mean()
                )
                feature_df[f'{indicator}_std_{period}'] = (
                    feature_df[indicator].rolling(period).std()
                )

            # Change features
            feature_df[f'{indicator}_change_1d'] = feature_df[indicator].diff()
            feature_df[f'{indicator}_change_5d'] = feature_df[indicator].diff(5)

        return feature_df

    # Additional methods to add to BigQueryClient for hourly data support

    def get_latest_datetime(self, table_name: str, ticker: str = None) -> Optional[datetime]:
        """
        Get the latest datetime for a ticker in hourly tables.

        Args:
            table_name: Name of the table (without project/dataset prefix)
            ticker: Stock ticker (optional)

        Returns:
            Latest datetime or None if no data exists
        """
        table_ref = BQ_TABLES.get(table_name)
        if not table_ref:
            raise ValueError(f"Unknown table: {table_name}")

        query = f"""
        SELECT MAX(datetime) as latest_datetime
        FROM `{table_ref}`
        """

        if ticker:
            query += f" WHERE ticker = '{ticker}'"

        try:
            result = self.query(query)
            if not result.empty and result['latest_datetime'].iloc[0] is not None:
                return pd.to_datetime(result['latest_datetime'].iloc[0])
            return None
        except Exception as e:
            logger.error(f"Error getting latest datetime: {e}")
            return None


    def create_hourly_tables(self):
        """Create BigQuery tables for hourly data."""
        # Create raw OHLCV hourly table
        raw_ohlcv_hourly_schema = [
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("datetime", "DATETIME", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("hour", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("volume", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("adjusted_close", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                                 default_value_expression="CURRENT_TIMESTAMP()"),
        ]

        # Create technical indicators hourly table
        indicators_hourly_schema = [
            bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("datetime", "DATETIME", mode="REQUIRED"),
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("hour", "INTEGER", mode="REQUIRED"),
            # Short-term hourly indicators
            bigquery.SchemaField("sma_20h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_140h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("sma_350h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_12h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_84h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ema_182h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rsi_14h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rsi_98h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_signal", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("macd_histogram", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_upper_20h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_middle_20h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("bb_lower_20h", "FLOAT64", mode="NULLABLE"),
            # Volume indicators
            bigquery.SchemaField("volume_sma_20h", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("volume_ratio", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("obv", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("vwap", "FLOAT64", mode="NULLABLE"),
            # Hourly-specific indicators
            bigquery.SchemaField("hour_of_day", "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("is_market_open", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_first_hour", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("is_last_hour", "BOOLEAN", mode="NULLABLE"),
            bigquery.SchemaField("high_of_day", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("low_of_day", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("pct_from_high", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("pct_from_low", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED",
                                 default_value_expression="CURRENT_TIMESTAMP()"),
        ]

        # Create tables with partitioning and clustering
        tables_to_create = [
            ("raw_ohlcv_hourly", raw_ohlcv_hourly_schema),
            ("technical_indicators_hourly", indicators_hourly_schema),
        ]

        for table_name, schema in tables_to_create:
            table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
            table = bigquery.Table(table_id, schema=schema)

            # Set partitioning on date column
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="date"
            )

            # Set clustering
            if table_name == "raw_ohlcv_hourly":
                table.clustering_fields = ["ticker", "hour"]
            elif table_name == "technical_indicators_hourly":
                table.clustering_fields = ["ticker", "hour"]

            try:
                table = self.client.create_table(table, exists_ok=True)
                logger.info(f"Created hourly table: {table_id}")
            except Exception as e:
                logger.error(f"Error creating hourly table {table_name}: {e}")

    def create_daily_aggregates_view(self):
        """Create a view that aggregates hourly data to daily."""
        view_id = f"{self.project_id}.{self.dataset_id}.daily_ohlcv_from_hourly"

        view_query = f"""
        CREATE OR REPLACE VIEW `{view_id}` AS
        WITH daily_data AS (
            SELECT 
                ticker,
                date,
                FIRST_VALUE(open) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as open,
                MAX(high) OVER (PARTITION BY ticker, date) as high,
                MIN(low) OVER (PARTITION BY ticker, date) as low,
                LAST_VALUE(close) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as close,
                SUM(volume) OVER (PARTITION BY ticker, date) as volume,
                LAST_VALUE(adjusted_close) OVER (
                    PARTITION BY ticker, date 
                    ORDER BY datetime
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) as adjusted_close,
                COUNT(*) OVER (PARTITION BY ticker, date) as hourly_count
            FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
            WHERE hour BETWEEN 9 AND 15  -- Regular trading hours only
        )
        SELECT DISTINCT
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume,
            adjusted_close,
            hourly_count
        FROM daily_data
        WHERE hourly_count >= 6  -- At least 6 hours of data
        """

        try:
            self.client.query(view_query).result()
            logger.info(f"Created daily aggregates view: {view_id}")
        except Exception as e:
            logger.error(f"Error creating daily aggregates view: {e}")


    def get_hourly_data_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get statistics about hourly data coverage."""
        query = f"""
        SELECT 
            ticker,
            COUNT(DISTINCT date) as trading_days,
            COUNT(*) as total_hours,
            AVG(volume) as avg_hourly_volume,
            MIN(datetime) as first_hour,
            MAX(datetime) as last_hour,
            COUNT(DISTINCT hour) as unique_hours,
            SUM(CASE WHEN hour BETWEEN 9 AND 15 THEN 1 ELSE 0 END) as regular_hours,
            SUM(CASE WHEN hour < 9 OR hour > 15 THEN 1 ELSE 0 END) as extended_hours
        FROM `{BQ_TABLES['raw_ohlcv_hourly']}`
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY ticker
        ORDER BY ticker
        """

        return self.query(query)