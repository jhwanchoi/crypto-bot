import sys
from unittest.mock import AsyncMock, Mock


class TestTelegramNotifierInit:
    """Tests for TelegramNotifier initialization."""

    def test_init_with_enabled_true_and_valid_credentials(self, monkeypatch):
        """Test initialization with enabled=True creates bot instance."""
        mock_telegram = Mock()
        mock_bot = Mock()
        mock_telegram.Bot.return_value = mock_bot
        monkeypatch.setitem(sys.modules, "telegram", mock_telegram)

        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=True
        )

        assert notifier.enabled is True
        assert notifier.bot_token == "test-token-123"
        assert notifier.chat_id == "test-chat-456"
        assert notifier._bot == mock_bot
        mock_telegram.Bot.assert_called_once_with(token="test-token-123")

    def test_init_with_enabled_true_but_import_fails(self, monkeypatch):
        """Test initialization handles telegram import failure gracefully."""
        import builtins

        # Remove telegram from sys.modules if it exists
        monkeypatch.delitem(sys.modules, "telegram", raising=False)

        # Mock __import__ to raise ImportError for telegram
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "telegram":
                raise ImportError("telegram module not found")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=True
        )

        assert notifier.enabled is True
        assert notifier._bot is None

    def test_init_with_enabled_false(self):
        """Test initialization with enabled=False does not create bot."""
        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=False
        )

        assert notifier.enabled is False
        assert notifier._bot is None


class TestTelegramNotifierSyncSend:
    """Tests for TelegramNotifier send_sync method."""

    def test_send_sync_when_enabled_success(self, monkeypatch):
        """Test send_sync successfully sends message when enabled."""
        mock_telegram = Mock()
        mock_bot = Mock()

        # Mock the async send_message
        async def mock_send_message(chat_id, text):
            return None

        mock_bot.send_message = AsyncMock(side_effect=mock_send_message)
        mock_telegram.Bot.return_value = mock_bot
        monkeypatch.setitem(sys.modules, "telegram", mock_telegram)

        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=True
        )

        result = notifier.send_sync("Sync test message")

        assert result is True
        mock_bot.send_message.assert_called_once()

    def test_send_sync_exception_handling(self, monkeypatch):
        """Test send_sync handles exceptions gracefully."""
        mock_telegram = Mock()
        mock_bot = Mock()

        # Make send_message raise an exception
        async def mock_send_message_error(chat_id, text):
            raise Exception("Connection failed")

        mock_bot.send_message = AsyncMock(side_effect=mock_send_message_error)
        mock_telegram.Bot.return_value = mock_bot
        monkeypatch.setitem(sys.modules, "telegram", mock_telegram)

        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=True
        )

        result = notifier.send_sync("Test message")

        assert result is False

    def test_send_sync_when_bot_is_none(self, monkeypatch):
        """Test send_sync returns False when bot is None."""
        import builtins

        # Remove telegram from sys.modules
        monkeypatch.delitem(sys.modules, "telegram", raising=False)

        # Mock __import__ to raise ImportError
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "telegram":
                raise ImportError("telegram module not found")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=True
        )

        result = notifier.send_sync("Test message")

        assert result is False

    def test_send_sync_when_disabled(self):
        """Test send_sync returns False when disabled."""
        from src.notification.telegram import TelegramNotifier

        notifier = TelegramNotifier(
            bot_token="test-token-123", chat_id="test-chat-456", enabled=False
        )

        result = notifier.send_sync("Test message")

        assert result is False
