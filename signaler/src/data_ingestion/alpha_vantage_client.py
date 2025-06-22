"""
Alpha Vantage API client for fetching financial data.
"""
import time
from typing import Dict, List, Optional, Union
import pandas as pd
import requests
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import aiohttp
from functools import lru_cache

from config.settings import (
    ALPHA_VANTAGE_API_KEY,
    ALPHA_VANTAGE_BASE_URL,
    ALPHA_VANTAGE_RATE_LIMIT,
    ALPHA_VANTAGE_TIMEOUT,
    MACRO_INDICATORS
)


class AlphaVantageClient:
    """Client for Alpha Vantage API with rate limiting and error handling."""

    def __init__(self, api_key: str = ALPHA_VANTAGE_API_KEY):
        """Initialize Alpha Vantage client."""
        self.api_key = api_key
        self.base_url = ALPHA_VANTAGE_BASE_URL
        self.rate_limit = ALPHA_VANTAGE_RATE_LIMIT
        self.timeout = ALPHA_VANTAGE_TIMEOUT
        self.last_request_time = 0
        self._session = None

    def _rate_limit_wait(self) -> None:
        """Implement rate limiting to avoid API throttling."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60 / self.rate_limit  # seconds between requests

        if time_since_last_request < min_interval:
            wait_time = min_interval - time_since_last_request
            logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """Make API request with error handling."""
        params['apikey'] = self.api_key

        self._rate_limit_wait()

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()

            logger.debug(f"ticker data: {data}")

            # Check for API errors
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                raise ValueError("API call frequency limit reached")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_daily_ohlcv(
            self,
            symbol: str,
            outputsize: str = 'full'
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data for a symbol."""
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        # Parse time series data
        time_series = data.get('Time Series (Daily)', {})

        if not time_series:
            raise ValueError(f"No data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df.columns = [
            'open', 'high', 'low', 'close',
            'adjusted_close', 'volume', 'dividend', 'split_coefficient'
        ]

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add ticker column
        df['ticker'] = symbol
        df['date'] = df.index.date

        # Select required columns
        return df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']]

    def get_intraday_data(
            self,
            symbol: str,
            interval: str = '5min',
            outputsize: str = 'full'
    ) -> pd.DataFrame:
        """Fetch intraday data for a symbol."""
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'datatype': 'json'
        }

        data = self._make_request(params)

        # Parse time series data
        time_series_key = f'Time Series ({interval})'
        time_series = data.get(time_series_key, {})

        if not time_series:
            raise ValueError(f"No intraday data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add ticker column
        df['ticker'] = symbol
        df['datetime'] = df.index

        return df

    def get_technical_indicator(
            self,
            symbol: str,
            indicator: str,
            interval: str = 'daily',
            time_period: int = None,
            series_type: str = 'close',
            **kwargs
    ) -> pd.DataFrame:
        """Fetch technical indicator data."""
        # Map indicator names to Alpha Vantage function names
        indicator_map = {
            'sma': 'SMA',
            'ema': 'EMA',
            'rsi': 'RSI',
            'macd': 'MACD',
            'bbands': 'BBANDS',
            'adx': 'ADX',
            'atr': 'ATR',
            'obv': 'OBV',
            'vwap': 'VWAP'
        }

        function_name = indicator_map.get(indicator.lower())
        if not function_name:
            raise ValueError(f"Unsupported indicator: {indicator}")

        params = {
            'function': function_name,
            'symbol': symbol,
            'interval': interval,
            'series_type': series_type
        }

        # Add time period if required
        if time_period and indicator.lower() in ['sma', 'ema', 'rsi', 'atr', 'adx']:
            params['time_period'] = time_period

        # Add any additional parameters
        params.update(kwargs)

        data = self._make_request(params)

        # Find the technical analysis key
        ta_key = None
        for key in data.keys():
            if 'Technical Analysis' in key:
                ta_key = key
                break

        if not ta_key:
            raise ValueError(f"No technical indicator data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data[ta_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add metadata
        df['ticker'] = symbol
        df['date'] = df.index.date
        df['indicator'] = indicator

        return df

    def get_economic_indicator(self, indicator: str) -> pd.DataFrame:
        """Fetch economic indicator data."""
        # Map economic indicators to Alpha Vantage functions
        indicator_map = {
            'gdp': 'REAL_GDP',
            'cpi': 'CPI',
            'inflation': 'INFLATION',
            'retail_sales': 'RETAIL_SALES',
            'unemployment_rate': 'UNEMPLOYMENT',
            'fed_funds_rate': 'FEDERAL_FUNDS_RATE',
            'treasury_yield': 'TREASURY_YIELD',
            'consumer_sentiment': 'CONSUMER_SENTIMENT',
            'nfp': 'NONFARM_PAYROLL',
            'wti_crude': 'WTI',
            'brent_crude': 'BRENT'
        }

        function_name = indicator_map.get(indicator.lower())
        if not function_name:
            logger.warning(f"Unsupported economic indicator: {indicator}")
            return pd.DataFrame()

        params = {
            'function': function_name,
            'datatype': 'json'
        }

        # Add specific parameters for certain indicators
        if indicator.lower() == 'treasury_yield':
            params['maturity'] = '10year'

        try:
            data = self._make_request(params)

            # Parse data based on indicator type
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                # Handle different response formats
                for key in data.keys():
                    if isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    return pd.DataFrame()

            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
                df = df.drop('timestamp', axis=1)

            # Add indicator name
            df['indicator'] = indicator

            return df

        except Exception as e:
            logger.error(f"Failed to fetch {indicator}: {e}")
            return pd.DataFrame()

    def get_sentiment_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch news sentiment data for multiple tickers."""
        all_sentiment = []

        for ticker in tickers:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'limit': 200
            }

            try:
                data = self._make_request(params)

                if 'feed' not in data:
                    logger.warning(f"No sentiment data for {ticker}")
                    continue

                # Process each news item
                for item in data['feed']:
                    # Extract ticker-specific sentiment
                    ticker_sentiment = None
                    for ts in item.get('ticker_sentiment', []):
                        if ts['ticker'] == ticker:
                            ticker_sentiment = ts
                            break

                    if ticker_sentiment:
                        sentiment_row = {
                            'ticker': ticker,
                            'date': pd.to_datetime(item['time_published']).date(),
                            'sentiment_score': float(ticker_sentiment['ticker_sentiment_score']),
                            'relevance_score': float(ticker_sentiment['relevance_score']),
                            'sentiment_label': ticker_sentiment['ticker_sentiment_label'],
                            'source': item.get('source', 'unknown'),
                            'title': item.get('title', ''),
                            'url': item.get('url', '')
                        }
                        all_sentiment.append(sentiment_row)

            except Exception as e:
                logger.error(f"Failed to fetch sentiment for {ticker}: {e}")
                continue

        if all_sentiment:
            df = pd.DataFrame(all_sentiment)

            # Aggregate by date
            agg_df = df.groupby(['ticker', 'date']).agg({
                'sentiment_score': 'mean',
                'relevance_score': 'mean',
                'source': 'count'  # count as volume_mentions
            }).reset_index()

            agg_df.rename(columns={'source': 'volume_mentions'}, inplace=True)

            return agg_df

        return pd.DataFrame()

    async def fetch_multiple_tickers_async(
            self,
            tickers: List[str],
            data_type: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """Asynchronously fetch data for multiple tickers."""
        async def fetch_ticker(session, ticker):
            try:
                if data_type == 'daily':
                    return ticker, self.get_daily_ohlcv(ticker)
                elif data_type == 'sentiment':
                    return ticker, self.get_sentiment_data([ticker])
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                return ticker, pd.DataFrame()

        # Create tasks for all tickers
        results = {}

        # Process in batches to respect rate limits
        batch_size = self.rate_limit
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            # Fetch batch synchronously (due to rate limits)
            for ticker in batch:
                if data_type == 'daily':
                    results[ticker] = self.get_daily_ohlcv(ticker)
                elif data_type == 'sentiment':
                    results[ticker] = self.get_sentiment_data([ticker])

                # Small delay between requests
                time.sleep(0.5)

        return results

    @lru_cache(maxsize=128)
    def get_sector_performance(self) -> pd.DataFrame:
        """Fetch sector performance data."""
        params = {
            'function': 'SECTOR'
        }

        data = self._make_request(params)

        # Extract real-time performance
        realtime = data.get('Rank A: Real-Time Performance', {})

        sectors = []
        for sector, performance in realtime.items():
            sectors.append({
                'sector': sector,
                'performance': float(performance.strip('%')) / 100,
                'date': pd.Timestamp.now().date()
            })

        return pd.DataFrame(sectors)

    def validate_api_key(self) -> bool:
        """Validate API key by making a test request."""
        try:
            # Make a simple request
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL'
            }
            data = self._make_request(params)
            return 'Global Quote' in data
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False