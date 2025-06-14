"""
Tests for data ingestion components.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data_ingestion.ohlcv_fetcher import OHLCVFetcher
from src.data_ingestion.alpha_vantage_client import AlphaVantageClient
from src.data_ingestion.data_validator import DataValidator
from src.data_ingestion.macro_data_fetcher import MacroDataFetcher


class TestAlphaVantageClient:
    """Test Alpha Vantage client functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return AlphaVantageClient(api_key="test_key")

    @pytest.fixture
    def sample_ohlcv_response(self):
        """Sample API response for OHLCV data."""
        return {
            "Meta Data": {
                "1. Information": "Daily Time Series with Splits and Dividend Events",
                "2. Symbol": "AAPL",
                "3. Last Refreshed": "2024-01-15",
                "4. Output Size": "Compact",
                "5. Time Zone": "US/Eastern"
            },
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "150.00",
                    "2. high": "152.00",
                    "3. low": "149.50",
                    "4. close": "151.50",
                    "5. adjusted close": "151.50",
                    "6. volume": "50000000",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0"
                },
                "2024-01-12": {
                    "1. open": "149.00",
                    "2. high": "150.50",
                    "3. low": "148.50",
                    "4. close": "150.00",
                    "5. adjusted close": "150.00",
                    "6. volume": "45000000",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0"
                }
            }
        }

    def test_parse_daily_ohlcv(self, client, sample_ohlcv_response):
        """Test parsing of daily OHLCV data."""
        with patch.object(client, '_make_request', return_value=sample_ohlcv_response):
            df = client.get_daily_ohlcv('AAPL')

            assert not df.empty
            assert len(df) == 2
            assert 'ticker' in df.columns
            assert df['ticker'].iloc[0] == 'AAPL'
            assert df['close'].iloc[0] == 151.50
            assert df['volume'].iloc[0] == 50000000

    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        client.last_request_time = time.time()

        with patch('time.sleep') as mock_sleep:
            client._rate_limit_wait()
            mock_sleep.assert_called_once()

    def test_error_handling(self, client):
        """Test API error handling."""
        error_response = {"Error Message": "Invalid API key"}

        with patch.object(client, '_make_request', return_value=error_response):
            with pytest.raises(ValueError, match="API Error"):
                client.get_daily_ohlcv('AAPL')


class TestDataValidator:
    """Test data validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return DataValidator()

    @pytest.fixture
    def valid_ohlcv_data(self):
        """Create valid OHLCV data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(110, 115, 10),
            'low': np.random.uniform(95, 100, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.randint(1000000, 5000000, 10)
        })

    @pytest.fixture
    def invalid_ohlcv_data(self):
        """Create invalid OHLCV data."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [100, 102, 104, 106, 108],
            'high': [105, 107, 103, 110, 112],  # high < low for index 2
            'low': [98, 100, 105, 104, 106],
            'close': [102, 104, 106, 108, 110],
            'volume': [1000000, 2000000, -1000, 3000000, 0]  # negative volume
        })

    def test_validate_ohlcv_valid(self, validator, valid_ohlcv_data):
        """Test validation of valid OHLCV data."""
        is_valid, issues = validator.validate_ohlcv(valid_ohlcv_data)

        assert is_valid
        assert len(issues) == 0

    def test_validate_ohlcv_invalid(self, validator, invalid_ohlcv_data):
        """Test validation of invalid OHLCV data."""
        is_valid, issues = validator.validate_ohlcv(invalid_ohlcv_data)

        assert not is_valid
        assert len(issues) > 0
        assert any('price_consistency' in issue for issue in issues)
        assert any('volume_validity' in issue for issue in issues)

    def test_check_price_consistency(self, validator, invalid_ohlcv_data):
        """Test price consistency check."""
        is_valid, issue = validator._check_price_consistency(invalid_ohlcv_data)

        assert not is_valid
        assert 'invalid OHLC relationships' in issue

    def test_check_volume_validity(self, validator, invalid_ohlcv_data):
        """Test volume validity check."""
        is_valid, issue = validator._check_volume_validity(invalid_ohlcv_data)

        assert not is_valid
        assert 'negative volume' in issue


class TestOHLCVFetcher:
    """Test OHLCV fetcher functionality."""

    @pytest.fixture
    def fetcher(self):
        """Create test fetcher."""
        with patch('src.data_ingestion.ohlcv_fetcher.AlphaVantageClient'):
            with patch('src.data_ingestion.ohlcv_fetcher.BigQueryClient'):
                return OHLCVFetcher()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        return pd.DataFrame({
            'ticker': ['AAPL'] * 5,
            'date': pd.date_range('2024-01-01', periods=5, freq='D'),
            'open': [150, 151, 152, 153, 154],
            'high': [152, 153, 154, 155, 156],
            'low': [149, 150, 151, 152, 153],
            'close': [151, 152, 153, 154, 155],
            'volume': [50000000, 52000000, 51000000, 53000000, 54000000],
            'adjusted_close': [151, 152, 153, 154, 155]
        })

    def test_fetch_historical_data(self, fetcher, sample_data):
        """Test historical data fetching."""
        fetcher.av_client.get_daily_ohlcv = Mock(return_value=sample_data)

        result = fetcher.fetch_historical_data(
            start_date='2024-01-01',
            end_date='2024-01-05',
            tickers=['AAPL']
        )

        assert 'AAPL' in result
        assert len(result['AAPL']) == 5
        assert result['AAPL']['close'].iloc[-1] == 155

    def test_store_to_bigquery(self, fetcher, sample_data):
        """Test storing data to BigQuery."""
        fetcher.bq_client.insert_dataframe = Mock()

        result = fetcher.store_to_bigquery({'AAPL': sample_data})

        assert result['AAPL'] == True
        fetcher.bq_client.insert_dataframe.assert_called_once()

    def test_backfill_missing_data(self, fetcher):
        """Test backfilling missing data."""
        # Mock BigQuery query results
        existing_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='D')
        })

        fetcher.bq_client.query = Mock(return_value=existing_data)
        fetcher._fetch_ticker_data = Mock(return_value=pd.DataFrame())

        result = fetcher.backfill_missing_data(check_window_days=7)

        assert isinstance(result, dict)


class TestMacroDataFetcher:
    """Test macro data fetcher functionality."""

    @pytest.fixture
    def fetcher(self):
        """Create test fetcher."""
        with patch('src.data_ingestion.macro_data_fetcher.AlphaVantageClient'):
            with patch('src.data_ingestion.macro_data_fetcher.BigQueryClient'):
                return MacroDataFetcher()

    def test_process_indicator_data(self, fetcher):
        """Test processing of indicator data."""
        raw_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=12, freq='M'),
            'value': np.random.uniform(100, 110, 12)
        })
        raw_data['indicator'] = 'gdp'

        processed = fetcher._process_indicator_data(raw_data, 'gdp')

        assert 'yoy_change' in processed.columns
        assert 'ma_3' in processed.columns
        assert 'trend_up' in processed.columns

    def test_calculate_composite_indicators(self, fetcher):
        """Test composite indicator calculation."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'gdp': np.random.uniform(100, 110, 10),
            'unemployment_rate': np.random.uniform(3, 5, 10),
            'cpi': np.random.uniform(100, 105, 10),
            'consumer_confidence': np.random.uniform(90, 110, 10)
        })

        result = fetcher.calculate_composite_indicators(data)

        assert 'economic_health_index' in result.columns

    def test_aggregate_to_wide_format(self, fetcher):
        """Test aggregation to wide format."""
        indicator_data = {
            'gdp': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=3, freq='Q'),
                'value': [100, 102, 104]
            }),
            'cpi': pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=3, freq='M'),
                'value': [100, 100.5, 101]
            })
        }

        result = fetcher.aggregate_to_wide_format(indicator_data)

        assert 'gdp' in result.columns
        assert 'cpi' in result.columns
        assert len(result) > 0


# Integration tests
class TestIntegration:
    """Integration tests for data ingestion pipeline."""

    @pytest.mark.integration
    def test_full_ingestion_pipeline(self):
        """Test full data ingestion pipeline."""
        # This would test the complete flow from API to BigQuery
        # Requires actual API keys and BigQuery access
        pass

    @pytest.mark.integration
    def test_daily_update_flow(self):
        """Test daily update workflow."""
        # This would test the daily update job
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])