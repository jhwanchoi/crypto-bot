import numpy as np
import pandas as pd

from src.strategy.indicators import (
    calculate_bollinger_bands,
    calculate_price_change,
    calculate_rsi,
    get_bb_position,
)


def test_rsi_range():
    """Test RSI values are within 0-100 range."""
    prices = pd.Series([100 + i * 2 for i in range(30)])
    rsi = calculate_rsi(prices, period=14)

    # Filter out NaN values
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0).all()
    assert (valid_rsi <= 100).all()


def test_rsi_oversold():
    """Test RSI indicates oversold on declining prices."""
    # Strong downtrend
    prices = pd.Series([100 - i * 2 for i in range(30)])
    rsi = calculate_rsi(prices, period=14)

    # Last RSI should be low (oversold)
    assert rsi.iloc[-1] < 40


def test_rsi_overbought():
    """Test RSI indicates overbought on rising prices."""
    # Strong uptrend
    prices = pd.Series([100 + i * 2 for i in range(30)])
    rsi = calculate_rsi(prices, period=14)

    # Last RSI should be high (overbought)
    assert rsi.iloc[-1] > 60


def test_rsi_neutral():
    """Test RSI around 50 for sideways movement."""
    # Oscillating prices
    prices = pd.Series([100 + (i % 2) * 2 for i in range(30)])
    rsi = calculate_rsi(prices, period=14)

    # Should be near neutral
    assert 30 < rsi.iloc[-1] < 70


def test_bollinger_bands_structure():
    """Test Bollinger Bands structure (upper >= middle >= lower)."""
    prices = pd.Series([100 + np.sin(i / 5) * 10 for i in range(30)])
    upper, middle, lower = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

    # Check non-NaN values
    for i in range(len(prices)):
        if not pd.isna(upper.iloc[i]):
            assert upper.iloc[i] >= middle.iloc[i]
            assert middle.iloc[i] >= lower.iloc[i]


def test_bollinger_bands_width():
    """Test Bollinger Bands width increases with volatility."""
    # Low volatility
    stable_prices = pd.Series([100 + i * 0.1 for i in range(30)])
    upper1, _, lower1 = calculate_bollinger_bands(stable_prices, period=20)
    width1 = upper1.iloc[-1] - lower1.iloc[-1]

    # High volatility
    volatile_prices = pd.Series([100 + i * 2 for i in range(30)])
    upper2, _, lower2 = calculate_bollinger_bands(volatile_prices, period=20)
    width2 = upper2.iloc[-1] - lower2.iloc[-1]

    assert width2 > width1


def test_get_bb_position_above_upper():
    """Test position detection above upper band."""
    position = get_bb_position(close=110, upper=105, lower=95)
    assert position == "above_upper"


def test_get_bb_position_below_lower():
    """Test position detection below lower band."""
    position = get_bb_position(close=90, upper=105, lower=95)
    assert position == "below_lower"


def test_get_bb_position_upper_half():
    """Test position detection in upper half."""
    position = get_bb_position(close=103, upper=105, lower=95)
    assert position == "upper_half"


def test_get_bb_position_lower_half():
    """Test position detection in lower half."""
    position = get_bb_position(close=97, upper=105, lower=95)
    assert position == "lower_half"


def test_get_bb_position_unknown():
    """Test position detection with NaN values."""
    position = get_bb_position(close=100, upper=np.nan, lower=95)
    assert position == "unknown"

    position = get_bb_position(close=100, upper=105, lower=np.nan)
    assert position == "unknown"


def test_calculate_price_change_positive():
    """Test positive price change calculation."""
    prices = pd.Series([100, 105, 110, 115, 120])
    change = calculate_price_change(prices, periods=4)

    expected = (120 - 100) / 100
    assert abs(change - expected) < 1e-10


def test_calculate_price_change_negative():
    """Test negative price change calculation."""
    prices = pd.Series([120, 115, 110, 105, 100])
    change = calculate_price_change(prices, periods=4)

    expected = (100 - 120) / 120
    assert abs(change - expected) < 1e-10


def test_calculate_price_change_insufficient_data():
    """Test price change with insufficient data."""
    prices = pd.Series([100, 105])
    change = calculate_price_change(prices, periods=5)

    assert change == 0.0


def test_bollinger_bands_middle_is_sma():
    """Test that middle band equals simple moving average."""
    prices = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118] * 3)
    _, middle, _ = calculate_bollinger_bands(prices, period=10)

    # Calculate SMA manually
    sma = prices.rolling(window=10).mean()

    # Compare non-NaN values
    valid_indices = ~middle.isna()
    assert np.allclose(middle[valid_indices], sma[valid_indices])
