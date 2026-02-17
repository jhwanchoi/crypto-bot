from src.notification.telegram import TelegramNotifier


def test_telegram_format_trade_message():
    """Test trade message formatting contains expected fields."""
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=False)

    msg = notifier.format_trade_message(
        ticker="KRW-BTC",
        side="buy",
        price=45000000.0,
        amount=0.001,
        total_krw=50000.0,
        rsi=28.5,
        reasoning="RSI oversold",
    )

    assert "KRW-BTC" in msg
    assert "45,000,000" in msg
    assert "0.001" in msg
    assert "50,000" in msg
    assert "28.5" in msg
    assert "RSI oversold" in msg
    assert "매수" in msg


def test_telegram_format_error_message():
    """Test error message formatting."""
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=False)

    msg = notifier.format_error_message("Connection timeout")

    assert "⚠️" in msg
    assert "Connection timeout" in msg


def test_telegram_format_status_message():
    """Test portfolio status message formatting."""
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=False)

    portfolio = {
        "KRW-BTC": {"amount": 0.001, "avg_price": 45000000},
        "KRW-ETH": {"amount": 0.05, "avg_price": 3000000},
    }

    msg = notifier.format_status_message(portfolio)

    assert "포트폴리오" in msg
    assert "KRW-BTC" in msg
    assert "KRW-ETH" in msg
    assert "0.001" in msg


def test_telegram_send_sync_disabled():
    """Test send_sync returns False when disabled."""
    notifier = TelegramNotifier(bot_token="", chat_id="", enabled=False)

    result = notifier.send_sync("test message")

    assert result is False
