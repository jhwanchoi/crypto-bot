from unittest.mock import patch

import pytest

from src.exchange.upbit_client import UpbitClient


@pytest.fixture
def client():
    """Create a paper trading client."""
    with patch("src.exchange.upbit_client.pyupbit.Upbit"):
        return UpbitClient(access_key="test", secret_key="test", paper_mode=True)


def test_initial_balance(client):
    """Test initial paper trading balance."""
    assert client.get_balance("KRW") == 500000.0
    assert client.get_balance("BTC") == 0.0


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_paper_buy(mock_price, client):
    """Test paper trading buy order."""
    mock_price.return_value = 50000000.0  # 50M KRW per BTC

    initial_krw = client.get_balance("KRW")
    result = client.buy_market_order("KRW-BTC", 100000.0)

    assert result["side"] == "bid"
    assert result["price"] == 50000000.0
    assert client.get_balance("KRW") == initial_krw - 100000.0

    # Check position was created
    btc_amount = client.get_position_amount("BTC")
    assert btc_amount > 0
    # Amount should be (100000 - fee) / price
    expected = (100000.0 - 100000.0 * 0.0005) / 50000000.0
    assert abs(btc_amount - expected) < 1e-10


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_paper_sell(mock_price, client):
    """Test paper trading sell order."""
    mock_price.return_value = 50000000.0

    # Buy first
    client.buy_market_order("KRW-BTC", 100000.0)
    btc_amount = client.get_position_amount("BTC")
    krw_before_sell = client.get_balance("KRW")

    # Sell all
    result = client.sell_market_order("KRW-BTC", btc_amount)

    assert result["side"] == "ask"
    assert result["price"] == 50000000.0
    assert client.get_position_amount("BTC") == 0.0

    # Should get back close to original amount (minus fees)
    krw_after = client.get_balance("KRW")
    assert krw_after > krw_before_sell


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_avg_buy_price(mock_price, client):
    """Test average buy price calculation."""
    # Buy at 50M
    mock_price.return_value = 50000000.0
    client.buy_market_order("KRW-BTC", 100000.0)

    avg1 = client.get_avg_buy_price("BTC")
    assert abs(avg1 - 50000000.0) < 100  # Within 100 KRW tolerance

    # Buy more at 60M
    mock_price.return_value = 60000000.0
    client.buy_market_order("KRW-BTC", 100000.0)

    avg2 = client.get_avg_buy_price("BTC")
    # Average should be between 50M and 60M
    assert 50000000.0 < avg2 < 60000000.0


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_partial_sell(mock_price, client):
    """Test selling only part of position."""
    mock_price.return_value = 50000000.0
    client.buy_market_order("KRW-BTC", 200000.0)

    initial_amount = client.get_position_amount("BTC")

    # Sell half
    client.sell_market_order("KRW-BTC", initial_amount / 2)

    remaining = client.get_position_amount("BTC")
    assert abs(remaining - initial_amount / 2) < 1e-10


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_buy_with_no_price(mock_price, client):
    """Test buy when price cannot be fetched."""
    mock_price.return_value = None

    result = client.buy_market_order("KRW-BTC", 100000.0)
    assert "error" in result


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_sell_more_than_owned(mock_price, client):
    """Test selling more than owned amount."""
    mock_price.return_value = 50000000.0
    client.buy_market_order("KRW-BTC", 100000.0)

    btc_amount = client.get_position_amount("BTC")

    # Try to sell 10x more
    result = client.sell_market_order("KRW-BTC", btc_amount * 10)

    # Should only sell what we have
    assert result["amount"] <= btc_amount
    assert client.get_position_amount("BTC") == 0.0


def test_live_mode_init():
    """Test live mode initialization creates pyupbit.Upbit instance."""
    with patch("src.exchange.upbit_client.pyupbit.Upbit") as mock_upbit:
        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        assert not client.paper_mode
        mock_upbit.assert_called_once_with("live_key", "live_secret")


def test_live_get_balance():
    """Test live mode get_balance calls upbit.get_balance."""
    with patch("src.exchange.upbit_client.pyupbit.Upbit") as mock_upbit_class:
        mock_upbit_instance = mock_upbit_class.return_value
        mock_upbit_instance.get_balance.return_value = 1000000.0

        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        balance = client.get_balance("KRW")

        assert balance == 1000000.0
        mock_upbit_instance.get_balance.assert_called_once_with("KRW")


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_live_get_current_price(mock_get_price):
    """Test live mode get_current_price calls pyupbit.get_current_price."""
    mock_get_price.return_value = 50000000.0

    with patch("src.exchange.upbit_client.pyupbit.Upbit"):
        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        price = client.get_current_price("KRW-BTC")

        assert price == 50000000.0
        mock_get_price.assert_called_once_with("KRW-BTC")


@patch("src.exchange.upbit_client.pyupbit.get_ohlcv")
def test_live_get_ohlcv(mock_get_ohlcv):
    """Test live mode get_ohlcv calls pyupbit.get_ohlcv."""
    import pandas as pd

    mock_df = pd.DataFrame({"close": [50000000.0] * 14})
    mock_get_ohlcv.return_value = mock_df

    with patch("src.exchange.upbit_client.pyupbit.Upbit"):
        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        ohlcv = client.get_ohlcv("KRW-BTC", interval="minute240", count=14)

        assert ohlcv is not None
        assert len(ohlcv) == 14
        mock_get_ohlcv.assert_called_once_with("KRW-BTC", interval="minute240", count=14)


def test_live_buy_market_order():
    """Test live mode buy_market_order calls upbit.buy_market_order."""
    with patch("src.exchange.upbit_client.pyupbit.Upbit") as mock_upbit_class:
        mock_upbit_instance = mock_upbit_class.return_value
        mock_upbit_instance.buy_market_order.return_value = {"uuid": "live-order-123"}

        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        result = client.buy_market_order("KRW-BTC", 100000.0)

        assert result["uuid"] == "live-order-123"
        mock_upbit_instance.buy_market_order.assert_called_once_with("KRW-BTC", 100000.0)


def test_live_sell_market_order():
    """Test live mode sell_market_order calls upbit.sell_market_order."""
    with patch("src.exchange.upbit_client.pyupbit.Upbit") as mock_upbit_class:
        mock_upbit_instance = mock_upbit_class.return_value
        mock_upbit_instance.sell_market_order.return_value = {"uuid": "live-sell-123"}

        client = UpbitClient(access_key="live_key", secret_key="live_secret", paper_mode=False)
        result = client.sell_market_order("KRW-BTC", 0.001)

        assert result["uuid"] == "live-sell-123"
        mock_upbit_instance.sell_market_order.assert_called_once_with("KRW-BTC", 0.001)


@patch("src.exchange.upbit_client.pyupbit.get_current_price")
def test_paper_sell_with_none_price(mock_price, client):
    """Test paper sell when price is None."""
    mock_price.return_value = None

    result = client.sell_market_order("KRW-BTC", 0.001)
    assert "error" in result
    assert result["error"] == "Cannot get current price"
