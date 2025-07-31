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
        self.rate_limit = ALPHA_VANTAGE_RATE_LIMIT  # Should be 75 for premium
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
        
        logger.debug(f"Making API request with params: {params}")
        
        self._rate_limit_wait()

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            logger.debug(f"Response status: {response.status_code}")
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Response data keys: {list(data.keys())}")

            # Check for API errors
            if "Error Message" in data:
                logger.error(f"API Error for params {params}: {data['Error Message']}")
                raise ValueError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note for params {params}: {data['Note']}")
                raise ValueError("API call frequency limit reached")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for params {params}: {e}")
            raise

    def get_daily_ohlcv(
            self,
            symbol: str,
            outputsize: str = 'full'
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data for a symbol."""
        logger.info(f"Fetching daily OHLCV data for {symbol} with outputsize={outputsize}")
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize
        }

        data = self._make_request(params)

        # Parse time series data
        time_series = data.get('Time Series (Daily)', {})
        
        logger.debug(f"Time series data found for {symbol}: {len(time_series)} days")
        
        if not time_series:
            logger.error(f"No time series data in response for {symbol}", 
                        symbol=symbol,
                        response_keys=list(data.keys()),
                        has_meta_data='Meta Data' in data,
                        has_information='Information' in data,
                        has_error='Error Message' in data,
                        has_note='Note' in data)
            
            # Check for common API response patterns
            if 'Meta Data' in data:
                logger.info(f"Meta data found for {symbol}", symbol=symbol, meta_data=data['Meta Data'])
            if 'Information' in data:
                logger.warning(f"API Information message for {symbol}", symbol=symbol, information=data['Information'])
            if 'Error Message' in data:
                logger.error(f"API Error Message for {symbol}", symbol=symbol, error_message=data['Error Message'])
            if 'Note' in data:
                logger.warning(f"API Note for {symbol}", symbol=symbol, note=data['Note'])
                
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

    def get_hourly_ohlcv(
            self,
            symbol: str,
            month: str = None,
            extended_hours: bool = True
    ) -> pd.DataFrame:
        """
        Fetch hourly OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            month: Specific month in YYYY-MM format (None for last 30 days)
            extended_hours: Include pre/post market data

        Returns:
            DataFrame with hourly OHLCV data
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': '60min',
            'adjusted': 'true',
            'extended_hours': str(extended_hours).lower(),
            'outputsize': 'full'
        }

        # Add month parameter if specified
        if month:
            params['month'] = month

        data = self._make_request(params)

        # Parse time series data
        time_series = data.get('Time Series (60min)', {})

        if not time_series:
            raise ValueError(f"No hourly data returned for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add metadata columns
        df['ticker'] = symbol
        df['datetime'] = df.index
        df['date'] = df.index.date
        df['hour'] = df.index.hour

        # For now, set adjusted_close same as close (Alpha Vantage doesn't provide hourly adjusted)
        df['adjusted_close'] = df['close']

        return df[['ticker', 'datetime', 'date', 'hour', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']]

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

    def get_economic_indicator(self, indicator: str) -> pd.DataFrame:
        """Fetch economic indicator data."""
        # Map common names to API function names
        function_map = {
            'gdp': 'REAL_GDP',
            'cpi': 'CPI',
            'unemployment': 'UNEMPLOYMENT',
            'fed_funds_rate': 'FEDERAL_FUNDS_RATE',
            'nonfarm_payroll': 'NONFARM_PAYROLL',
            'retail_sales': 'RETAIL_SALES',
            'durables': 'DURABLES',
            'inflation': 'INFLATION',
            'treasury_yield': 'TREASURY_YIELD',
            'consumer_sentiment': 'CONSUMER_SENTIMENT',
            'gdp_per_capita': 'REAL_GDP_PER_CAPITA',
            'pce': 'PCE',
            'inflation_expectation': 'INFLATION_EXPECTATION'
        }

        api_function = function_map.get(indicator, indicator.upper())

        params = {
            'function': api_function,
            'datatype': 'json'
        }

        # Some indicators need specific parameters
        if indicator == 'treasury_yield':
            params['maturity'] = 'monthly'

        data = self._make_request(params)

        # Extract data key (varies by indicator)
        data_key = None
        for key in data.keys():
            if 'data' in key.lower() or key == 'data':
                data_key = key
                break

        if not data_key:
            raise ValueError(f"No data key found for {indicator}")

        # Convert to DataFrame
        df = pd.DataFrame(data[data_key])

        # Standardize column names
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])

        # Get value column (varies by indicator)
        value_col = None
        for col in df.columns:
            if col not in ['date', 'timestamp']:
                value_col = col
                break

        if value_col:
            df['value'] = pd.to_numeric(df[value_col], errors='coerce')

        df['indicator'] = indicator

        # Select standard columns
        return df[['date', 'indicator', 'value']].sort_values('date')

    def get_sentiment_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch market sentiment data."""
        results = []

        for ticker in tickers:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'limit': 100
            }

            try:
                data = self._make_request(params)

                if 'feed' in data:
                    for article in data['feed']:
                        sentiment_data = {
                            'time_published': pd.to_datetime(article.get('time_published')),
                            'title': article.get('title'),
                            'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                            'overall_sentiment_label': article.get('overall_sentiment_label')
                        }

                        # Get ticker-specific sentiment
                        ticker_found = False
                        for ticker_sentiment in article.get('ticker_sentiment', []):
                            if ticker_sentiment.get('ticker') == ticker:
                                sentiment_data['ticker_sentiment_score'] = float(
                                    ticker_sentiment.get('ticker_sentiment_score', 0)
                                )
                                sentiment_data['ticker_sentiment_label'] = ticker_sentiment.get(
                                    'ticker_sentiment_label'
                                )
                                ticker_found = True
                                break
                        
                        # Only include articles that mention this specific ticker
                        if ticker_found or any(ticker in article.get('title', '') for ticker in [ticker]):
                            results.append(sentiment_data)

            except Exception as e:
                logger.error(f"Failed to fetch sentiment for {ticker}: {e}")

        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_multi_ticker_data(
            self,
            tickers: List[str],
            data_type: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        results = {}

        # Process in batches to respect rate limits
        for ticker in tickers:
            if data_type == 'daily':
                results[ticker] = self.get_daily_ohlcv(ticker)
            elif data_type == 'hourly':
                results[ticker] = self.get_hourly_ohlcv(ticker)
            elif data_type == 'sentiment':
                results[ticker] = self.get_sentiment_data([ticker])

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