"""
Temporal feature engineering for time-based patterns.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import holidays
from src.utils import logger
from config.settings import BQ_TABLES
from src.utils.bigquery import BigQueryClient


class TemporalFeatureEngineer:
    """Generate temporal features from dates."""

    def __init__(self, country: str = 'US'):
        """Initialize with country for holiday calendar."""
        self.country = country
        self.holidays = holidays.US()  # Can be extended to other countries
        self.bq_client = BigQueryClient()

        # Define earnings season months (typically Jan, Apr, Jul, Oct)
        self.earnings_months = [1, 4, 7, 10]

        # Market events calendar (can be extended)
        self.market_events = {
            'fomc_meetings': self._get_fomc_dates(),
            'options_expiry': self._get_options_expiry_dates(),
            'quarter_ends': self._get_quarter_end_dates()
        }

    def generate_temporal_features(
            self,
            start_date: str,
            end_date: str
    ) -> pd.DataFrame:
        """
        Generate comprehensive temporal features for date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with temporal features
        """
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame({'date': date_range})

        # Basic temporal features
        df = self._add_basic_features(df)

        # Holiday features
        df = self._add_holiday_features(df)

        # Trading day features
        df = self._add_trading_features(df)

        # Market event features
        df = self._add_market_event_features(df)

        # Cyclical encoding
        df = self._add_cyclical_features(df)

        # Seasonality features
        df = self._add_seasonality_features(df)

        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic date-based features."""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year

        # Day name and month name (for potential one-hot encoding)
        df['day_name'] = df['date'].dt.day_name()
        df['month_name'] = df['date'].dt.month_name()

        # Start/end of period indicators
        df['is_month_start'] = df['date'].dt.is_month_start
        df['is_month_end'] = df['date'].dt.is_month_end
        df['is_quarter_start'] = df['date'].dt.is_quarter_start
        df['is_quarter_end'] = df['date'].dt.is_quarter_end
        df['is_year_start'] = df['date'].dt.is_year_start
        df['is_year_end'] = df['date'].dt.is_year_end

        return df

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday-related features."""
        # Is holiday
        df['is_holiday'] = df['date'].apply(lambda x: x in self.holidays)

        # Days to/from nearest holiday
        df['days_to_next_holiday'] = df['date'].apply(self._days_to_next_holiday)
        df['days_from_last_holiday'] = df['date'].apply(self._days_from_last_holiday)

        # Specific holiday periods
        df['is_thanksgiving_week'] = (
                (df['month'] == 11) &
                (df['week_of_year'].isin([47, 48]))
        )

        df['is_christmas_period'] = (
                (df['month'] == 12) &
                (df['day_of_month'] >= 15)
        )

        df['is_new_year_period'] = (
                ((df['month'] == 12) & (df['day_of_month'] >= 26)) |
                ((df['month'] == 1) & (df['day_of_month'] <= 7))
        )

        # Summer vacation period
        df['is_summer_vacation'] = (
                (df['month'].isin([7, 8])) |
                ((df['month'] == 6) & (df['day_of_month'] >= 15))
        )

        return df

    def _add_trading_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading-specific features."""
        # Is trading day (not weekend or holiday)
        df['is_trading_day'] = (
                (df['day_of_week'] < 5) &  # Not weekend
                (~df['is_holiday'])  # Not holiday
        )

        # Trading days in month/quarter
        df['trading_days_in_month'] = df.groupby(
            [df['year'], df['month']]
        )['is_trading_day'].transform('sum')

        df['trading_days_in_quarter'] = df.groupby(
            [df['year'], df['quarter']]
        )['is_trading_day'].transform('sum')

        # Position in trading month/quarter
        df['trading_day_of_month'] = df.groupby(
            [df['year'], df['month']]
        )['is_trading_day'].cumsum()

        df['trading_day_of_quarter'] = df.groupby(
            [df['year'], df['quarter']]
        )['is_trading_day'].cumsum()

        # First/last trading days
        df['is_first_trading_day_of_month'] = (
                df['is_trading_day'] &
                (df['trading_day_of_month'] == 1)
        )

        df['is_last_trading_day_of_month'] = (
                df['is_trading_day'] &
                (df['trading_day_of_month'] == df['trading_days_in_month'])
        )

        # Triple/Quadruple witching (3rd Friday of Mar, Jun, Sep, Dec)
        df['is_triple_witching'] = (
                (df['month'].isin([3, 6, 9, 12])) &
                (df['day_of_week'] == 4) &  # Friday
                (df['day_of_month'] >= 15) &
                (df['day_of_month'] <= 21)
        )

        return df

    def _add_market_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for known market events."""
        # Earnings season
        df['is_earnings_season'] = df['month'].isin(self.earnings_months)

        # FOMC meetings (approximate - typically Tue/Wed of specific weeks)
        df['days_to_fomc'] = df['date'].apply(self._days_to_next_fomc)
        df['is_fomc_week'] = df['days_to_fomc'] <= 7

        # Options expiry (3rd Friday of each month)
        df['is_options_expiry'] = (
                (df['day_of_week'] == 4) &  # Friday
                (df['day_of_month'] >= 15) &
                (df['day_of_month'] <= 21)
        )

        # End of quarter rebalancing period
        df['is_quarter_end_rebalance'] = (
                df['is_quarter_end'] |
                (df['date'] + pd.Timedelta(days=5)).dt.is_quarter_end
        )

        # Tax loss harvesting period (November-December)
        df['is_tax_loss_period'] = df['month'].isin([11, 12])

        return df

    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encodings for periodic features."""
        # Day of week cyclical encoding
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Day of month cyclical encoding
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

        # Month cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Week of year cyclical encoding
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        return df

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonality and market regime features."""
        # Seasons
        df['season'] = df['month'].apply(self._get_season)

        # Market regimes/patterns
        df['is_january_effect'] = (df['month'] == 1)
        df['is_sell_in_may'] = df['month'].isin([5, 6, 7, 8, 9])
        df['is_santa_rally'] = (
                (df['month'] == 12) &
                (df['day_of_month'] >= 20)
        )

        # Window dressing periods (end of quarter/year)
        df['is_window_dressing'] = (
                (df['is_quarter_end']) |
                ((df['date'] + pd.Timedelta(days=5)).dt.is_quarter_end)
        )

        # Volume patterns
        df['is_low_volume_period'] = (
                df['is_summer_vacation'] |
                df['is_thanksgiving_week'] |
                df['is_christmas_period']
        )

        return df

    def _days_to_next_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days to next holiday."""
        for i in range(1, 366):
            future_date = date + timedelta(days=i)
            if future_date in self.holidays:
                return i
        return 365

    def _days_from_last_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days from last holiday."""
        for i in range(1, 366):
            past_date = date - timedelta(days=i)
            if past_date in self.holidays:
                return i
        return 365

    def _get_fomc_dates(self) -> List[pd.Timestamp]:
        """Get approximate FOMC meeting dates."""
        # This is a simplified version - in production, use actual FOMC calendar
        fomc_dates = []
        current_year = datetime.now().year

        # Approximate 8 meetings per year
        for year in range(current_year - 2, current_year + 2):
            for month in [1, 3, 5, 6, 7, 9, 11, 12]:
                # Usually around 2nd or 3rd Tuesday/Wednesday
                fomc_dates.append(pd.Timestamp(f'{year}-{month:02d}-15'))

        return fomc_dates

    def _days_to_next_fomc(self, date: pd.Timestamp) -> int:
        """Calculate days to next FOMC meeting."""
        future_meetings = [d for d in self.market_events['fomc_meetings'] if d > date]
        if future_meetings:
            return (future_meetings[0] - date).days
        return 999

    def _get_options_expiry_dates(self) -> List[pd.Timestamp]:
        """Get options expiry dates (3rd Friday of each month)."""
        expiry_dates = []
        current_year = datetime.now().year

        for year in range(current_year - 2, current_year + 2):
            for month in range(1, 13):
                # Find 3rd Friday
                first_day = pd.Timestamp(f'{year}-{month:02d}-01')
                first_friday = first_day + pd.Timedelta(days=(4 - first_day.dayofweek) % 7)
                third_friday = first_friday + pd.Timedelta(weeks=2)
                expiry_dates.append(third_friday)

        return expiry_dates

    def _get_quarter_end_dates(self) -> List[pd.Timestamp]:
        """Get quarter end dates."""
        quarter_ends = []
        current_year = datetime.now().year

        for year in range(current_year - 2, current_year + 2):
            for quarter in [3, 6, 9, 12]:
                quarter_ends.append(pd.Timestamp(f'{year}-{quarter:02d}-30'))

        return quarter_ends

    def _get_season(self, month: int) -> str:
        """Get season from month."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'

    def store_temporal_features(
            self,
            start_date: str,
            end_date: str
    ) -> bool:
        """Generate and store temporal features to BigQuery."""
        try:
            # Generate features
            features_df = self.generate_temporal_features(start_date, end_date)

            # Convert date to proper format for BigQuery
            features_df['date'] = features_df['date'].dt.date

            # Store to BigQuery
            self.bq_client.insert_dataframe(
                features_df,
                'temporal_features',
                if_exists='append'
            )

            logger.info(f"Stored {len(features_df)} temporal feature records")
            return True

        except Exception as e:
            logger.error("Failed to store temporal features")
            return False

    def get_feature_importance(self) -> pd.DataFrame:
        """Analyze importance of temporal features based on market returns."""
        query = f"""
        WITH market_returns AS (
            SELECT 
                date,
                AVG((close - open) / open) as daily_return,
                STDDEV((close - open) / open) as return_volatility
            FROM `{BQ_TABLES['raw_ohlcv']}`
            WHERE ticker IN ('SPY', 'QQQ', 'DIA')
            GROUP BY date
        )
        SELECT 
            tf.*,
            mr.daily_return,
            mr.return_volatility
        FROM `{BQ_TABLES['temporal_features']}` tf
        JOIN market_returns mr ON tf.date = mr.date
        """

        df = self.bq_client.query(query)

        # Calculate correlations with returns
        feature_cols = [col for col in df.columns
                        if col not in ['date', 'daily_return', 'return_volatility']]

        importance = {}
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                importance[col] = {
                    'return_correlation': df[col].corr(df['daily_return']),
                    'volatility_correlation': df[col].corr(df['return_volatility'])
                }

        return pd.DataFrame(importance).T