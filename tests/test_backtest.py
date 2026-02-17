import numpy as np
import pandas as pd

from backtest.backtester import Backtester


def test_backtester_returns_expected_keys():
    """Test backtester returns all expected result keys."""
    backtester = Backtester(
        initial_cash=500000.0,
        base_amount=5000.0,
        rsi_period=14,
        rsi_buy_threshold=30.0,
        rsi_sell_threshold=70.0,
    )

    # Generate synthetic price data
    prices = pd.Series(np.random.uniform(40000, 50000, 100))

    result = backtester.run(prices)

    # Check all expected keys are present
    assert "total_return_pct" in result
    assert "total_trades" in result
    assert "win_rate" in result
    assert "sharpe_ratio" in result
    assert "max_drawdown" in result
    assert "min_cash" in result
    assert "final_value" in result


def test_backtester_no_negative_cash():
    """Test backtester never goes negative on cash."""
    backtester = Backtester(
        initial_cash=10000.0,  # Small cash
        base_amount=5000.0,
        rsi_period=14,
        rsi_buy_threshold=30.0,
        rsi_sell_threshold=70.0,
    )

    prices = pd.Series(np.random.uniform(40000, 50000, 50))
    result = backtester.run(prices)

    assert result["min_cash"] >= 0


def test_backtester_handles_flat_prices():
    """Test backtester handles flat price series."""
    backtester = Backtester()

    # Flat prices (no RSI signals)
    prices = pd.Series([45000.0] * 50)
    result = backtester.run(prices)

    # Should not crash, returns valid result
    assert isinstance(result, dict)
    assert result["total_trades"] >= 0


def test_backtester_insufficient_data():
    """Test backtester handles insufficient data gracefully."""
    backtester = Backtester(rsi_period=14)

    # Only 10 prices (less than RSI period)
    prices = pd.Series([45000.0] * 10)
    result = backtester.run(prices)

    assert result["total_return_pct"] == 0.0
    assert result["total_trades"] == 0
    assert result["final_value"] == backtester.initial_cash


def test_backtester_buy_and_sell_execution():
    """Test backtest with prices that trigger both buy and sell signals."""
    backtester = Backtester(
        initial_cash=500000.0,
        base_amount=50000.0,
        rsi_period=14,
        rsi_buy_threshold=30.0,
        rsi_sell_threshold=70.0,
    )

    # Create price pattern that causes RSI to cross both thresholds
    # Start high, drop low (RSI < 30), then surge high (RSI > 70)
    prices_high = [50000.0] * 15  # Initial stable period
    prices_drop = [49000.0, 47000.0, 45000.0, 43000.0, 41000.0, 39000.0, 37000.0]  # Drop
    prices_surge = [
        38000.0,
        40000.0,
        43000.0,
        46000.0,
        49000.0,
        52000.0,
        55000.0,
        58000.0,
        61000.0,
    ]  # Surge

    all_prices = prices_high + prices_drop + prices_surge
    prices = pd.Series(all_prices)

    result = backtester.run(prices)

    # Should have executed both buy and sell trades
    assert result["total_trades"] > 0
    assert isinstance(result["win_rate"], float)
    assert result["sharpe_ratio"] is not None


def test_backtester_with_nan_rsi():
    """Test backtester skips candles with NaN RSI."""
    backtester = Backtester(
        initial_cash=500000.0, base_amount=50000.0, rsi_period=14, rsi_buy_threshold=30.0
    )

    # Create prices where some RSI values will be NaN
    prices = pd.Series([45000.0] * 20)
    result = backtester.run(prices)

    # Should complete without error even with NaN RSI values
    assert isinstance(result, dict)
    assert "total_trades" in result


def test_backtester_win_rate_calculation():
    """Test win rate calculation with profitable and unprofitable sells."""
    backtester = Backtester(
        initial_cash=1000000.0,
        base_amount=100000.0,
        rsi_period=14,
        rsi_buy_threshold=30.0,
        rsi_sell_threshold=70.0,
    )

    # Create pattern with multiple buy-sell cycles
    # Pattern: stable -> drop (buy) -> rise (sell) -> drop (buy) -> drop more (sell at loss)
    prices_stable = [50000.0] * 15
    prices_drop1 = [48000.0, 46000.0, 44000.0, 42000.0, 40000.0, 38000.0]  # Buy trigger
    prices_rise = [39000.0, 41000.0, 44000.0, 47000.0, 50000.0, 53000.0, 56000.0]  # Sell at profit
    prices_drop2 = [54000.0, 52000.0, 50000.0, 48000.0, 46000.0, 44000.0, 42000.0]  # Buy trigger
    prices_surge = [43000.0, 45000.0, 48000.0, 51000.0, 54000.0, 57000.0, 60000.0]  # Sell at profit

    all_prices = prices_stable + prices_drop1 + prices_rise + prices_drop2 + prices_surge
    prices = pd.Series(all_prices)

    result = backtester.run(prices)

    # Should have sell trades with win rate calculated
    assert result["total_trades"] > 0
    assert 0.0 <= result["win_rate"] <= 100.0


def test_backtester_sharpe_ratio_zero_std():
    """Test Sharpe ratio when all returns are identical (std = 0)."""
    backtester = Backtester(
        initial_cash=500000.0,
        base_amount=50000.0,
        rsi_period=14,
        rsi_buy_threshold=30.0,
        rsi_sell_threshold=70.0,
    )

    # Completely flat prices - no trades, zero variance
    prices = pd.Series([50000.0] * 30)
    result = backtester.run(prices)

    # Sharpe ratio should be 0.0 when std is 0
    assert result["sharpe_ratio"] == 0.0


def test_backtester_empty_equity_curve():
    """Test backtester handles empty equity curve gracefully."""
    backtester = Backtester(rsi_period=14)

    # Just enough data for RSI calculation but no trading
    prices = pd.Series([50000.0] * 15)
    result = backtester.run(prices)

    # Should handle empty or minimal equity curve
    assert result["max_drawdown"] >= 0.0
    assert isinstance(result["sharpe_ratio"], float)
