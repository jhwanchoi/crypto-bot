import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.db import TradeDB


def test_tradedb_init(tmp_path):
    """Test TradeDB initialization creates database and table."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    assert db_path.exists()

    # Verify table exists by querying it
    trades = db.get_recent_trades()
    assert trades == []


def test_insert_trade(tmp_path):
    """Test inserting a trade record."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    db.insert_trade(
        ticker="KRW-BTC",
        side="buy",
        price=50000000.0,
        amount=0.001,
        total_krw=50000.0,
        rsi=65.5,
        rl_signal="buy",
        llm_reasoning="Strong bullish trend",
    )

    trades = db.get_recent_trades(limit=1)
    assert len(trades) == 1

    trade = trades[0]
    assert trade["ticker"] == "KRW-BTC"
    assert trade["side"] == "buy"
    assert trade["price"] == 50000000.0
    assert trade["amount"] == 0.001
    assert trade["total_krw"] == 50000.0
    assert trade["rsi"] == 65.5
    assert trade["rl_signal"] == "buy"
    assert trade["llm_reasoning"] == "Strong bullish trend"


def test_insert_trade_with_optional_fields(tmp_path):
    """Test inserting a trade with optional fields as None."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    db.insert_trade(ticker="KRW-ETH", side="sell", price=3000000.0, amount=0.5, total_krw=1500000.0)

    trades = db.get_recent_trades(limit=1)
    assert len(trades) == 1

    trade = trades[0]
    assert trade["ticker"] == "KRW-ETH"
    assert trade["rsi"] is None
    assert trade["rl_signal"] is None
    assert trade["llm_reasoning"] is None


def test_get_recent_trades(tmp_path):
    """Test retrieving recent trades with limit."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    # Insert multiple trades
    for i in range(15):
        db.insert_trade(
            ticker=f"KRW-COIN{i}", side="buy", price=10000.0 + i, amount=1.0, total_krw=10000.0 + i
        )

    # Get last 10 trades
    trades = db.get_recent_trades(limit=10)
    assert len(trades) == 10

    # Verify ordering (most recent first)
    assert trades[0]["ticker"] == "KRW-COIN14"
    assert trades[9]["ticker"] == "KRW-COIN5"


def test_get_daily_trade_count(tmp_path):
    """Test counting today's trades."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    # Insert trades
    for _ in range(5):
        db.insert_trade(
            ticker="KRW-BTC", side="buy", price=50000000.0, amount=0.001, total_krw=50000.0
        )

    count = db.get_daily_trade_count()
    assert count == 5


def test_get_portfolio_summary_empty(tmp_path):
    """Test portfolio summary with no trades."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    summary = db.get_portfolio_summary()
    assert summary["total_invested"] == 0
    assert summary["total_trades"] == {}


def test_get_portfolio_summary_single_ticker(tmp_path):
    """Test portfolio summary with a single ticker."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    db.insert_trade(ticker="KRW-BTC", side="buy", price=50000000.0, amount=0.001, total_krw=50000.0)

    db.insert_trade(ticker="KRW-BTC", side="buy", price=51000000.0, amount=0.001, total_krw=51000.0)

    summary = db.get_portfolio_summary()
    assert summary["total_invested"] == 101000.0
    assert summary["total_trades"]["KRW-BTC"] == 2


def test_get_portfolio_summary_multiple_tickers(tmp_path):
    """Test portfolio summary with multiple tickers."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    # BTC trades
    db.insert_trade(ticker="KRW-BTC", side="buy", price=50000000.0, amount=0.001, total_krw=50000.0)

    db.insert_trade(
        ticker="KRW-BTC", side="sell", price=51000000.0, amount=0.001, total_krw=51000.0
    )

    # ETH trades
    db.insert_trade(ticker="KRW-ETH", side="buy", price=3000000.0, amount=0.5, total_krw=1500000.0)

    summary = db.get_portfolio_summary()
    assert summary["total_invested"] == 1550000.0  # Only buy orders count
    assert summary["total_trades"]["KRW-BTC"] == 2
    assert summary["total_trades"]["KRW-ETH"] == 1


def test_get_portfolio_summary_sell_orders(tmp_path):
    """Test that sell orders don't contribute to total_invested."""
    db_path = tmp_path / "test_trades.db"
    db = TradeDB(str(db_path))

    db.insert_trade(ticker="KRW-BTC", side="buy", price=50000000.0, amount=0.001, total_krw=50000.0)

    db.insert_trade(
        ticker="KRW-BTC", side="sell", price=51000000.0, amount=0.001, total_krw=51000.0
    )

    summary = db.get_portfolio_summary()
    assert summary["total_invested"] == 50000.0  # Only the buy order
    assert summary["total_trades"]["KRW-BTC"] == 2  # Both counted in trades
