from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.collector import MarketDataCollector
from src.data.sentiment import get_fear_greed_index


def test_get_fear_greed_index_success():
    """Test successful Fear & Greed Index fetch."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"value": "75"}]}
    mock_response.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_response):
        result = get_fear_greed_index()
        assert result == 75


def test_get_fear_greed_index_failure_fallback():
    """Test Fear & Greed Index fallback on failure."""
    with patch("requests.get", side_effect=Exception("Network error")):
        result = get_fear_greed_index()
        assert result == 50  # Fallback value


def test_market_data_collector_collect():
    """Test MarketDataCollector.collect returns expected structure."""
    # Mock exchange
    mock_exchange = MagicMock()
    mock_ohlcv = pd.DataFrame(
        {
            "close": [40000, 41000, 42000, 43000, 44000] + [45000] * 20,
            "open": [39000, 40000, 41000, 42000, 43000] + [44000] * 20,
            "high": [41000, 42000, 43000, 44000, 45000] + [46000] * 20,
            "low": [39000, 40000, 41000, 42000, 43000] + [44000] * 20,
            "volume": [100] * 25,
        }
    )
    mock_exchange.get_ohlcv.return_value = mock_ohlcv

    with patch("src.data.collector.get_fear_greed_index", return_value=60):
        collector = MarketDataCollector(mock_exchange)
        result = collector.collect("KRW-BTC", rsi_period=14)

        assert result is not None
        assert "ticker" in result
        assert "price" in result
        assert "rsi" in result
        assert "bb_position" in result
        assert "price_changes" in result
        assert "fear_greed" in result
        assert result["ticker"] == "KRW-BTC"
        assert result["fear_greed"] == 60


def test_market_data_collector_insufficient_data():
    """Test collector returns None when insufficient data."""
    mock_exchange = MagicMock()
    mock_exchange.get_ohlcv.return_value = None

    collector = MarketDataCollector(mock_exchange)
    result = collector.collect("KRW-BTC")

    assert result is None


def test_market_data_collector_exception_handling():
    """Test collector returns None when exchange raises exception."""
    mock_exchange = MagicMock()
    mock_exchange.get_ohlcv.side_effect = Exception("Network error")

    collector = MarketDataCollector(mock_exchange)
    result = collector.collect("KRW-BTC")

    assert result is None
