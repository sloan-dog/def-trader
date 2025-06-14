"""
Macro economic data fetcher for economic indicators.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import time

from config.settings import MACRO_INDICATORS, BQ_TABLES
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from src.utils.bigquery_client import BigQueryClient


class MacroDataFetcher:
    """Fetch macro economic indicators."""

    def __init__(self):
        """Initialize fetcher."""
        self.av_client = AlphaVantageClient()
        self.bq_client = BigQueryClient()

        # Map indicators to their update frequencies
        self.update_frequencies = {
            'gdp': 'quarterly',
            'cpi': 'monthly',
            'pce': 'monthly',
            'nfp': 'monthly',
            'unemployment_rate': 'monthly',
            'fed_funds_rate': 'daily',
            'yield_curve_spread': 'daily',
            'retail_sales': 'monthly',
            'ism_manufacturing': 'monthly',
            'ism_services': 'monthly',
            'consumer_confidence': 'monthly',
            'wti_crude': 'daily',
            'brent_crude': 'daily',
            'china_pmi': 'monthly',
            'china_gdp': 'quarterly',
            'm2_money_supply': 'monthly'
        }

    def fetch_all_indicators(self) -> Dict[str, pd.DataFrame]:
        """Fetch all configured macro indicators."""
        results = {}

        for indicator in MACRO_INDICATORS:
            try:
                logger.info(f"Fetching {indicator}")

                # Fetch from Alpha Vantage
                df = self.av_client.get_economic_indicator(indicator)

                if not df.empty:
                    # Process and standardize
                    df = self._process_indicator_data(df, indicator)
                    results[indicator] = df
                    logger.info(f"Fetched {len(df)} records for {indicator}")
                else:
                    logger.warning(f"No data returned for {indicator}")

                # Rate limiting
                time.sleep(12)  # 5 calls per minute limit

            except Exception as e:
                logger.error(f"Failed to fetch {indicator}: {e}")

        return results

    def _process_indicator_data(
            self,
            df: pd.DataFrame,
            indicator: str
    ) -> pd.DataFrame:
        """Process and standardize indicator data."""
        # Ensure date column
        if 'date' not in df.columns:
            logger.error(f"No date column for {indicator}")
            return pd.DataFrame()

        # Convert date
        df['date'] = pd.to_datetime(df['date'])

        # Rename value column
        value_cols = [col for col in df.columns if col not in ['date', 'indicator']]
        if value_cols:
            df['value'] = df[value_cols[0]]
            df = df[['date', 'value', 'indicator']]

        # Sort by date
        df = df.sort_values('date')

        # Calculate derived features
        df = self._calculate_derived_features(df, indicator)

        return df

    def _calculate_derived_features(
            self,
            df: pd.DataFrame,
            indicator: str
    ) -> pd.DataFrame:
        """Calculate derived features for indicators."""
        # Year-over-year change
        if self.update_frequencies.get(indicator) in ['monthly', 'quarterly']:
            periods = 12 if self.update_frequencies[indicator] == 'monthly' else 4
            df['yoy_change'] = df['value'].pct_change(periods)
            df['yoy_diff'] = df['value'].diff(periods)

        # Moving averages
        if len(df) > 3:
            df['ma_3'] = df['value'].rolling(3).mean()
        if len(df) > 12:
            df['ma_12'] = df['value'].rolling(12).mean()

        # Trend direction
        df['trend_up'] = (df['value'] > df['value'].shift(1)).astype(int)

        # Z-score (standardized value)
        if len(df) > 20:
            df['z_score'] = (
                    (df['value'] - df['value'].rolling(20).mean()) /
                    df['value'].rolling(20).std()
            )

        return df

    def aggregate_to_wide_format(
            self,
            indicator_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Aggregate indicators into wide format by date."""
        # Start with date range
        all_dates = set()
        for df in indicator_data.values():
            if not df.empty:
                all_dates.update(df['date'].tolist())

        if not all_dates:
            return pd.DataFrame()

        # Create base dataframe
        date_range = pd.date_range(
            start=min(all_dates),
            end=max(all_dates),
            freq='D'
        )
        result_df = pd.DataFrame({'date': date_range})

        # Merge each indicator
        for indicator, df in indicator_data.items():
            if df.empty:
                continue

            # Rename columns
            rename_cols = {
                'value': indicator,
                'yoy_change': f'{indicator}_yoy',
                'ma_3': f'{indicator}_ma3',
                'z_score': f'{indicator}_zscore'
            }

            indicator_df = df[['date'] + [col for col in rename_cols.keys() if col in df.columns]]
            indicator_df = indicator_df.rename(columns=rename_cols)

            # Merge
            result_df = result_df.merge(indicator_df, on='date', how='left')

            # Forward fill based on update frequency
            freq = self.update_frequencies.get(indicator, 'monthly')
            if freq == 'daily':
                result_df[indicator] = result_df[indicator].fillna(method='ffill', limit=1)
            elif freq == 'monthly':
                result_df[indicator] = result_df[indicator].fillna(method='ffill', limit=30)
            elif freq == 'quarterly':
                result_df[indicator] = result_df[indicator].fillna(method='ffill', limit=90)

        return result_df

    def calculate_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite economic indicators."""
        # Economic Health Index (simple composite)
        health_components = ['gdp', 'unemployment_rate', 'cpi', 'consumer_confidence']
        available_components = [col for col in health_components if col in df.columns]

        if len(available_components) >= 2:
            # Normalize components
            for col in available_components:
                df[f'{col}_norm'] = (
                        (df[col] - df[col].mean()) / df[col].std()
                )

            # Inverse unemployment for health index
            if 'unemployment_rate_norm' in df.columns:
                df['unemployment_rate_norm'] = -df['unemployment_rate_norm']

            # Calculate composite
            norm_cols = [f'{col}_norm' for col in available_components]
            df['economic_health_index'] = df[norm_cols].mean(axis=1)

            # Clean up
            df = df.drop(columns=norm_cols)

        # Yield curve indicator
        if 'yield_curve_spread' in df.columns:
            df['yield_curve_inverted'] = (df['yield_curve_spread'] < 0).astype(int)
            df['yield_curve_ma30'] = df['yield_curve_spread'].rolling(30).mean()

        # Inflation expectations
        if 'cpi' in df.columns and 'pce' in df.columns:
            df['inflation_avg'] = (df['cpi'] + df['pce']) / 2
            if 'cpi_yoy' in df.columns and 'pce_yoy' in df.columns:
                df['inflation_yoy_avg'] = (df['cpi_yoy'] + df['pce_yoy']) / 2

        # Oil price average
        if 'wti_crude' in df.columns and 'brent_crude' in df.columns:
            df['oil_price_avg'] = (df['wti_crude'] + df['brent_crude']) / 2
            df['oil_spread'] = df['brent_crude'] - df['wti_crude']

        # Global growth proxy
        if 'gdp' in df.columns and 'china_gdp' in df.columns:
            df['global_growth_proxy'] = (df['gdp'] + df['china_gdp']) / 2

        return df

    def store_to_bigquery(
            self,
            df: pd.DataFrame,
            table_name: str = 'macro_indicators'
    ) -> bool:
        """Store macro indicators to BigQuery."""
        try:
            # Convert date format
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Store to BigQuery
            self.bq_client.insert_dataframe(
                df,
                table_name,
                if_exists='append'
            )

            logger.info(f"Stored {len(df)} macro indicator records")
            return True

        except Exception as e:
            logger.error(f"Failed to store macro data: {e}")
            return False

    def get_latest_indicators(self) -> pd.DataFrame:
        """Get latest values for all indicators."""
        query = f"""
        WITH latest_dates AS (
            SELECT 
                MAX(date) as latest_date
            FROM `{BQ_TABLES['macro_indicators']}`
        ),
        recent_data AS (
            SELECT *
            FROM `{BQ_TABLES['macro_indicators']}`
            WHERE date >= DATE_SUB((SELECT latest_date FROM latest_dates), INTERVAL 90 DAY)
        )
        SELECT 
            'GDP' as indicator,
            gdp as value,
            gdp_yoy as yoy_change,
            MAX(date) as last_update
        FROM recent_data
        WHERE gdp IS NOT NULL
        GROUP BY gdp, gdp_yoy
        
        UNION ALL
        
        SELECT 
            'CPI' as indicator,
            cpi as value,
            cpi_yoy as yoy_change,
            MAX(date) as last_update
        FROM recent_data
        WHERE cpi IS NOT NULL
        GROUP BY cpi, cpi_yoy
        
        -- Add other indicators similarly
        """

        return self.bq_client.query(query)

    def check_data_freshness(self) -> Dict[str, Dict]:
        """Check freshness of macro data."""
        query = f"""
        SELECT 
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as date_count
        FROM `{BQ_TABLES['macro_indicators']}`
        """

        result = self.bq_client.query(query)

        freshness_report = {}
        if not result.empty:
            latest_date = pd.to_datetime(result['latest_date'].iloc[0])
            days_old = (datetime.now() - latest_date).days

            for indicator in MACRO_INDICATORS:
                # Check individual indicator freshness
                ind_query = f"""
                SELECT MAX(date) as latest_date
                FROM `{BQ_TABLES['macro_indicators']}`
                WHERE {indicator} IS NOT NULL
                """

                ind_result = self.bq_client.query(ind_query)
                if not ind_result.empty and ind_result['latest_date'].iloc[0]:
                    ind_latest = pd.to_datetime(ind_result['latest_date'].iloc[0])
                    ind_days_old = (datetime.now() - ind_latest).days

                    # Expected update frequency
                    freq = self.update_frequencies.get(indicator, 'monthly')
                    expected_days = {
                        'daily': 1,
                        'monthly': 30,
                        'quarterly': 90
                    }.get(freq, 30)

                    freshness_report[indicator] = {
                        'latest_date': ind_latest,
                        'days_old': ind_days_old,
                        'update_frequency': freq,
                        'needs_update': ind_days_old > expected_days * 1.5
                    }

        return freshness_report


class MacroDataUpdater:
    """Orchestrator for macro data updates."""

    def __init__(self):
        """Initialize updater."""
        self.fetcher = MacroDataFetcher()

    def run_full_update(self) -> bool:
        """Run full macro data update."""
        logger.info("Starting macro data update")

        try:
            # Check data freshness
            freshness = self.fetcher.check_data_freshness()

            # Determine which indicators need updating
            indicators_to_update = []
            for indicator, status in freshness.items():
                if status.get('needs_update', True):
                    indicators_to_update.append(indicator)

            if not indicators_to_update:
                logger.info("All macro indicators are up to date")
                return True

            logger.info(f"Updating {len(indicators_to_update)} indicators")

            # Fetch data
            indicator_data = self.fetcher.fetch_all_indicators()

            if indicator_data:
                # Aggregate to wide format
                wide_df = self.fetcher.aggregate_to_wide_format(indicator_data)

                # Calculate composite indicators
                wide_df = self.fetcher.calculate_composite_indicators(wide_df)

                # Store to BigQuery
                success = self.fetcher.store_to_bigquery(wide_df)

                if success:
                    logger.info("Macro data update completed successfully")
                    return True

            logger.error("Failed to fetch macro data")
            return False

        except Exception as e:
            logger.error(f"Macro data update failed: {e}")
            return False