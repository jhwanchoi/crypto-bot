from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.main import TradingBot


@patch("src.main.load_config")
@patch("src.main.setup_logger")
def test_trading_bot_init(mock_logger, mock_load_config):
    """Test TradingBot initializes without error in paper mode with RL/LLM disabled."""
    # Mock config
    mock_config = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
        "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
        "assets": [{"ticker": "KRW-BTC", "allocation": 1.0}],
        "strategy": {
            "type": "dca_rsi",
            "interval_hours": 4,
            "base_amount_krw": 5000,
            "rsi_period": 14,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        },
        "risk": {
            "max_buy_per_trade_pct": 5,
            "max_daily_trades": 6,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
            "take_profit_sell_ratio": 0.3,
            "max_position_pct": 80,
            "min_cash_pct": 20,
        },
        "rl": {"enabled": False},
        "llm": {"enabled": False},
        "notification": {"telegram": {"enabled": False}},
    }
    mock_load_config.return_value = mock_config
    mock_logger.return_value = MagicMock()

    # Initialize bot
    bot = TradingBot(config_path="config/test.yaml")

    # Verify initialization
    assert bot.paper_mode is True
    assert bot.rl_enabled is False
    assert bot.llm_enabled is False
    assert len(bot.assets) == 1
    assert bot.assets[0]["ticker"] == "KRW-BTC"
    assert bot.strategy.base_amount == 5000
    assert bot.risk_manager.config.max_daily_trades == 6


@pytest.fixture
def mock_bot():
    """Create a TradingBot instance with all dependencies mocked."""
    with (
        patch("src.main.load_config") as mock_load_config,
        patch("src.main.setup_logger") as mock_logger,
    ):
        mock_config = {
            "mode": "paper",
            "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
            "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
            "assets": [{"ticker": "KRW-BTC", "allocation": 1.0}],
            "strategy": {
                "type": "dca_rsi",
                "interval_hours": 4,
                "base_amount_krw": 5000,
                "rsi_period": 14,
                "rsi_buy_threshold": 30,
                "rsi_sell_threshold": 70,
                "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
            },
            "risk": {
                "max_buy_per_trade_pct": 5,
                "max_daily_trades": 6,
                "stop_loss_pct": 8,
                "take_profit_pct": 15,
                "take_profit_sell_ratio": 0.3,
                "max_position_pct": 80,
                "min_cash_pct": 20,
            },
            "rl": {"enabled": False},
            "llm": {"enabled": False},
            "notification": {"telegram": {"enabled": False}},
        }
        mock_load_config.return_value = mock_config
        mock_logger.return_value = MagicMock()

        bot = TradingBot(config_path="config/test.yaml")

        # Mock all components
        bot.collector = Mock()
        bot.exchange = Mock()
        bot.strategy = Mock()
        bot.risk_manager = Mock()
        bot.db = Mock()
        bot.notifier = Mock()
        bot.rl_agent = None
        bot.llm_advisor = None

        yield bot


def test_run_cycle_buy_path_success(mock_bot):
    """Test run_cycle executes buy when RSI is oversold and risk allows."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 25.0,  # oversold
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 30,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {
        "type": "buy",
        "amount": 5000,
        "reason": "RSI oversold",
    }
    mock_bot.exchange.get_balance.return_value = 100000
    mock_bot.db.get_daily_trade_count.return_value = 2
    mock_bot.exchange.get_current_price.return_value = 50000000.0
    mock_bot.risk_manager.check_buy_allowed.return_value = (True, "")
    mock_bot.exchange.buy_market_order.return_value = {"amount": 0.0001, "status": "ok"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.collector.collect.assert_called_once_with("KRW-BTC", rsi_period=14)
    mock_bot.strategy.determine_action.assert_called_once_with(25.0, has_position=False)
    mock_bot.risk_manager.check_buy_allowed.assert_called_once()
    mock_bot.exchange.buy_market_order.assert_called_once_with("KRW-BTC", 5000)
    mock_bot.db.insert_trade.assert_called_once()
    mock_bot.notifier.send_sync.assert_called_once()


def test_run_cycle_sell_path_success(mock_bot):
    """Test run_cycle executes sell when RSI is overbought and position exists."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 55000000.0,
        "rsi": 75.0,  # overbought
        "bb_position": "upper_half",
        "price_changes": {"5": 0.05, "10": 0.10, "20": 0.15},
        "fear_greed": 70,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0.001, 0.001]  # has position
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0.001,
        "reason": "RSI overbought",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 50000000.0
    mock_bot.risk_manager.should_stop_loss.return_value = False
    mock_bot.risk_manager.should_take_profit.return_value = (False, 0)
    mock_bot.exchange.sell_market_order.return_value = {"amount": 0.001, "status": "ok"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.collector.collect.assert_called_once()
    mock_bot.strategy.determine_action.assert_called_once_with(75.0, has_position=True)
    mock_bot.exchange.sell_market_order.assert_called_once_with("KRW-BTC", 0.001)
    mock_bot.db.insert_trade.assert_called_once()
    mock_bot.notifier.send_sync.assert_called_once()


def test_run_cycle_skip_path(mock_bot):
    """Test run_cycle skips when RSI is in neutral range."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 52000000.0,
        "rsi": 50.0,  # neutral
        "bb_position": "middle",
        "price_changes": {"5": 0.01, "10": 0.02, "20": 0.03},
        "fear_greed": 50,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {"type": "skip", "reason": "RSI neutral"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.collector.collect.assert_called_once()
    mock_bot.strategy.determine_action.assert_called_once()
    mock_bot.exchange.buy_market_order.assert_not_called()
    mock_bot.exchange.sell_market_order.assert_not_called()
    mock_bot.db.insert_trade.assert_not_called()


def test_run_cycle_data_collection_failure(mock_bot):
    """Test run_cycle handles data collection failure gracefully."""
    # Setup mocks
    mock_bot.collector.collect.return_value = None

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.collector.collect.assert_called_once()
    mock_bot.strategy.determine_action.assert_not_called()
    mock_bot.exchange.buy_market_order.assert_not_called()
    mock_bot.notifier.send_sync.assert_called_once()


def test_run_cycle_buy_blocked_by_risk(mock_bot):
    """Test run_cycle blocks buy when risk manager denies it."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 25.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 30,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {
        "type": "buy",
        "amount": 50000,
        "reason": "RSI oversold",
    }
    mock_bot.exchange.get_balance.return_value = 10000
    mock_bot.db.get_daily_trade_count.return_value = 10  # exceeded max
    mock_bot.exchange.get_current_price.return_value = 50000000.0
    mock_bot.risk_manager.check_buy_allowed.return_value = (
        False,
        "Daily trade limit exceeded",
    )

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.risk_manager.check_buy_allowed.assert_called_once()
    mock_bot.exchange.buy_market_order.assert_not_called()
    mock_bot.db.insert_trade.assert_not_called()


def test_run_cycle_buy_order_failure(mock_bot):
    """Test run_cycle handles buy order failure from exchange."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 25.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 30,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {
        "type": "buy",
        "amount": 5000,
        "reason": "RSI oversold",
    }
    mock_bot.exchange.get_balance.return_value = 100000
    mock_bot.db.get_daily_trade_count.return_value = 2
    mock_bot.exchange.get_current_price.return_value = 50000000.0
    mock_bot.risk_manager.check_buy_allowed.return_value = (True, "")
    mock_bot.exchange.buy_market_order.return_value = {
        "error": "insufficient balance",
    }

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.exchange.buy_market_order.assert_called_once()
    mock_bot.db.insert_trade.assert_not_called()


def test_run_cycle_sell_order_failure(mock_bot):
    """Test run_cycle handles sell order failure from exchange."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 55000000.0,
        "rsi": 75.0,
        "bb_position": "upper_half",
        "price_changes": {"5": 0.05, "10": 0.10, "20": 0.15},
        "fear_greed": 70,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0.001, 0.001]
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0.001,
        "reason": "RSI overbought",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 50000000.0
    mock_bot.risk_manager.should_stop_loss.return_value = False
    mock_bot.risk_manager.should_take_profit.return_value = (False, 0)
    mock_bot.exchange.sell_market_order.return_value = {"error": "insufficient amount"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.exchange.sell_market_order.assert_called_once()
    mock_bot.db.insert_trade.assert_not_called()


def test_run_cycle_sell_with_stop_loss(mock_bot):
    """Test run_cycle triggers stop-loss when price drops too much."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 46000000.0,  # dropped 8% from 50M
        "rsi": 45.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.08, "10": -0.10, "20": -0.12},
        "fear_greed": 40,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0.001, 0.001]
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0.001,
        "reason": "RSI signal",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 50000000.0
    mock_bot.risk_manager.should_stop_loss.return_value = True
    mock_bot.risk_manager.should_take_profit.return_value = (False, 0)
    mock_bot.exchange.sell_market_order.return_value = {"amount": 0.001, "status": "ok"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.risk_manager.should_stop_loss.assert_called_once_with(50000000.0, 46000000.0)
    mock_bot.exchange.sell_market_order.assert_called_once()


def test_run_cycle_sell_with_take_profit(mock_bot):
    """Test run_cycle triggers partial take-profit when price rises enough."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 58000000.0,  # up 16% from 50M
        "rsi": 65.0,
        "bb_position": "upper_half",
        "price_changes": {"5": 0.08, "10": 0.12, "20": 0.16},
        "fear_greed": 75,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0.001, 0.001]
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0.001,
        "reason": "RSI overbought",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 50000000.0
    mock_bot.risk_manager.should_stop_loss.return_value = False
    mock_bot.risk_manager.should_take_profit.return_value = (True, 0.3)  # sell 30%
    mock_bot.exchange.sell_market_order.return_value = {"amount": 0.0003, "status": "ok"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.risk_manager.should_take_profit.assert_called_once_with(50000000.0, 58000000.0)
    mock_bot.exchange.sell_market_order.assert_called_once_with("KRW-BTC", 0.0003)


def test_run_cycle_sell_no_position(mock_bot):
    """Test run_cycle skips sell when no position exists."""
    # Setup mocks
    mock_bot.collector.collect.return_value = {
        "price": 55000000.0,
        "rsi": 75.0,
        "bb_position": "upper_half",
        "price_changes": {"5": 0.05, "10": 0.10, "20": 0.15},
        "fear_greed": 70,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0, 0]  # no position
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0,
        "reason": "RSI overbought",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 0

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.exchange.sell_market_order.assert_not_called()
    mock_bot.db.insert_trade.assert_not_called()


def test_run_cycle_llm_override_skip(mock_bot):
    """Test run_cycle respects LLM override to skip trading."""
    # Setup mocks
    mock_bot.llm_enabled = True
    mock_bot.llm_advisor = Mock()
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 28.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 30,
    }
    mock_bot.llm_advisor.analyze.return_value = {
        "action_override": "skip",
        "reasoning": "Market uncertainty too high",
        "market_phase": "uncertain",
    }
    mock_bot.exchange.get_position_amount.return_value = 0

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.llm_advisor.analyze.assert_called_once()
    mock_bot.strategy.determine_action.assert_not_called()
    mock_bot.exchange.buy_market_order.assert_not_called()


def test_run_cycle_llm_override_sell_all(mock_bot):
    """Test run_cycle respects LLM override to sell all."""
    # Setup mocks
    mock_bot.llm_enabled = True
    mock_bot.llm_advisor = Mock()
    mock_bot.collector.collect.return_value = {
        "price": 52000000.0,
        "rsi": 50.0,
        "bb_position": "middle",
        "price_changes": {"5": 0.0, "10": 0.01, "20": 0.02},
        "fear_greed": 50,
    }
    mock_bot.llm_advisor.analyze.return_value = {
        "action_override": "sell_all",
        "reasoning": "Major risk event detected",
        "market_phase": "danger",
    }
    mock_bot.exchange.get_position_amount.return_value = 0.001

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.llm_advisor.analyze.assert_called_once()
    mock_bot.strategy.determine_action.assert_not_called()


def test_run_cycle_rl_signal_applied(mock_bot):
    """Test run_cycle applies RL agent signal overrides to strategy."""
    # Setup mocks
    mock_bot.rl_enabled = True
    mock_bot.rl_agent = Mock()
    mock_bot.rl_agent.is_available = True
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 35.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 35,
    }

    rl_prediction = Mock()
    rl_prediction.action = "buy"
    rl_prediction.confidence = 0.85
    rl_prediction.suggested_params = {
        "rsi_buy_threshold": 35,
        "buy_multiplier": 1.3,
    }
    mock_bot.rl_agent.predict.return_value = rl_prediction

    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {"type": "skip", "reason": "neutral"}

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.rl_agent.predict.assert_called_once()
    call_args = mock_bot.rl_agent.predict.call_args[0][0]
    assert isinstance(call_args, np.ndarray)
    mock_bot.strategy.apply_overrides.assert_called_once_with(
        {"rsi_buy_threshold": 35, "buy_multiplier": 1.3}
    )


def test_run_cycle_general_exception(mock_bot):
    """Test run_cycle handles general exceptions and notifies."""
    # Setup mocks
    mock_bot.collector.collect.side_effect = Exception("Network error")

    # Execute
    mock_bot.run_cycle("KRW-BTC", 1.0)

    # Verify
    mock_bot.notifier.send_sync.assert_called_once()
    mock_bot.notifier.format_error_message.assert_called_once()


def test_run_iterates_all_assets(mock_bot):
    """Test run method iterates over all configured assets."""
    # Setup mocks
    mock_bot.assets = [
        {"ticker": "KRW-BTC", "allocation": 0.6},
        {"ticker": "KRW-ETH", "allocation": 0.4},
    ]
    mock_bot.run_cycle = Mock()

    # Execute
    mock_bot.run()

    # Verify
    assert mock_bot.run_cycle.call_count == 2
    mock_bot.run_cycle.assert_any_call("KRW-BTC", 0.6)
    mock_bot.run_cycle.assert_any_call("KRW-ETH", 0.4)


def test_start_scheduler_setup(mock_bot):
    """Test start method sets up scheduler and runs immediately."""
    mock_bot.run = Mock()

    with patch("src.main.BlockingScheduler") as mock_scheduler_class:
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        mock_scheduler.start.side_effect = KeyboardInterrupt()

        # Execute
        mock_bot.start()

        # Verify
        mock_scheduler_class.assert_called_once()
        mock_scheduler.add_job.assert_called_once_with(mock_bot.run, "interval", hours=4)
        mock_bot.run.assert_called_once()
        mock_scheduler.start.assert_called_once()


@patch("src.main.load_config")
@patch("src.main.setup_logger")
def test_trading_bot_init_with_rl_enabled(mock_logger, mock_load_config):
    """Test TradingBot initializes RL agent when enabled."""
    mock_config = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
        "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
        "assets": [{"ticker": "KRW-BTC", "allocation": 1.0}],
        "strategy": {
            "interval_hours": 4,
            "base_amount_krw": 5000,
            "rsi_period": 14,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        },
        "risk": {
            "max_buy_per_trade_pct": 5,
            "max_daily_trades": 6,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
            "take_profit_sell_ratio": 0.3,
            "max_position_pct": 80,
            "min_cash_pct": 20,
        },
        "rl": {"enabled": True, "model_path": "models/test.zip", "fallback_to_default": True},
        "llm": {"enabled": False},
        "notification": {"telegram": {"enabled": False}},
    }
    mock_load_config.return_value = mock_config

    with patch("src.main.FinRLAgent") as mock_finrl:
        mock_agent = Mock()
        mock_agent.is_available = False
        mock_finrl.return_value = mock_agent

        bot = TradingBot(config_path="config/test.yaml")
        assert bot.rl_enabled is True
        assert bot.rl_agent is not None
        mock_finrl.assert_called_once_with(model_path="models/test.zip", fallback=True)


@patch("src.main.load_config")
@patch("src.main.setup_logger")
def test_trading_bot_init_rl_failure(mock_logger, mock_load_config):
    """Test TradingBot handles RL agent init failure gracefully."""
    mock_config = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
        "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
        "assets": [],
        "strategy": {
            "interval_hours": 4,
            "base_amount_krw": 5000,
            "rsi_period": 14,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        },
        "risk": {
            "max_buy_per_trade_pct": 5,
            "max_daily_trades": 6,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
            "take_profit_sell_ratio": 0.3,
            "max_position_pct": 80,
            "min_cash_pct": 20,
        },
        "rl": {"enabled": True, "model_path": "bad_path", "fallback_to_default": True},
        "llm": {"enabled": False},
        "notification": {"telegram": {"enabled": False}},
    }
    mock_load_config.return_value = mock_config

    with patch("src.main.FinRLAgent", side_effect=Exception("RL init failed")):
        bot = TradingBot(config_path="config/test.yaml")
        assert bot.rl_enabled is True
        assert bot.rl_agent is None


@patch("src.main.load_config")
@patch("src.main.setup_logger")
def test_trading_bot_init_with_llm_enabled(mock_logger, mock_load_config):
    """Test TradingBot initializes LLM advisor when enabled."""
    mock_config = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
        "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
        "assets": [],
        "strategy": {
            "interval_hours": 4,
            "base_amount_krw": 5000,
            "rsi_period": 14,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        },
        "risk": {
            "max_buy_per_trade_pct": 5,
            "max_daily_trades": 6,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
            "take_profit_sell_ratio": 0.3,
            "max_position_pct": 80,
            "min_cash_pct": 20,
        },
        "rl": {"enabled": False},
        "llm": {
            "enabled": True,
            "model_id": "anthropic.claude-opus-4-6-v1",
            "region": "us-east-1",
            "temperature": 0,
            "max_tokens": 1024,
            "aws_profile": "prod",
            "verify_ssl": False,
        },
        "notification": {"telegram": {"enabled": False}},
    }
    mock_load_config.return_value = mock_config

    with patch("src.main.LLMAdvisor") as mock_llm:
        mock_advisor = Mock()
        mock_llm.return_value = mock_advisor

        bot = TradingBot(config_path="config/test.yaml")
        assert bot.llm_enabled is True
        assert bot.llm_advisor is not None
        mock_llm.assert_called_once_with(
            model_id="anthropic.claude-opus-4-6-v1",
            region="us-east-1",
            enabled=True,
            temperature=0,
            max_tokens=1024,
            profile_name="prod",
            verify_ssl=False,
        )


@patch("src.main.load_config")
@patch("src.main.setup_logger")
def test_trading_bot_init_llm_failure(mock_logger, mock_load_config):
    """Test TradingBot handles LLM advisor init failure gracefully."""
    mock_config = {
        "mode": "paper",
        "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
        "exchange": {"name": "upbit", "access_key": "", "secret_key": ""},
        "assets": [],
        "strategy": {
            "interval_hours": 4,
            "base_amount_krw": 5000,
            "rsi_period": 14,
            "rsi_buy_threshold": 30,
            "rsi_sell_threshold": 70,
            "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        },
        "risk": {
            "max_buy_per_trade_pct": 5,
            "max_daily_trades": 6,
            "stop_loss_pct": 8,
            "take_profit_pct": 15,
            "take_profit_sell_ratio": 0.3,
            "max_position_pct": 80,
            "min_cash_pct": 20,
        },
        "rl": {"enabled": False},
        "llm": {"enabled": True, "model_id": "test"},
        "notification": {"telegram": {"enabled": False}},
    }
    mock_load_config.return_value = mock_config

    with patch("src.main.LLMAdvisor", side_effect=Exception("LLM init failed")):
        bot = TradingBot(config_path="config/test.yaml")
        assert bot.llm_enabled is True
        assert bot.llm_advisor is None


def test_run_cycle_rl_prediction_failure(mock_bot):
    """Test run_cycle handles RL prediction failure gracefully."""
    mock_bot.rl_enabled = True
    mock_bot.rl_agent = Mock()
    mock_bot.rl_agent.is_available = True
    mock_bot.rl_agent.predict.side_effect = Exception("RL predict error")

    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 35.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 35,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {"type": "skip", "reason": "neutral"}

    mock_bot.run_cycle("KRW-BTC", 1.0)

    # RL failed but cycle continued
    mock_bot.strategy.determine_action.assert_called_once()


def test_run_cycle_llm_analysis_failure(mock_bot):
    """Test run_cycle handles LLM analysis failure gracefully."""
    mock_bot.llm_enabled = True
    mock_bot.llm_advisor = Mock()
    mock_bot.llm_advisor.analyze.side_effect = Exception("LLM error")

    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 35.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 35,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {"type": "skip", "reason": "neutral"}

    mock_bot.run_cycle("KRW-BTC", 1.0)

    # LLM failed but cycle continued
    mock_bot.strategy.determine_action.assert_called_once()


def test_run_cycle_buy_execution_exception(mock_bot):
    """Test run_cycle handles buy execution exception."""
    mock_bot.collector.collect.return_value = {
        "price": 50000000.0,
        "rsi": 25.0,
        "bb_position": "lower_half",
        "price_changes": {"5": -0.02, "10": -0.05, "20": -0.08},
        "fear_greed": 30,
    }
    mock_bot.exchange.get_position_amount.return_value = 0
    mock_bot.strategy.determine_action.return_value = {
        "type": "buy",
        "amount": 5000,
        "reason": "RSI oversold",
    }
    mock_bot.exchange.get_balance.return_value = 100000
    mock_bot.db.get_daily_trade_count.return_value = 2
    mock_bot.exchange.get_current_price.return_value = 50000000.0
    mock_bot.risk_manager.check_buy_allowed.return_value = (True, "")
    mock_bot.exchange.buy_market_order.side_effect = Exception("Network timeout")

    mock_bot.run_cycle("KRW-BTC", 1.0)

    mock_bot.notifier.send_sync.assert_called_once()
    mock_bot.notifier.format_error_message.assert_called_once()


def test_run_cycle_sell_execution_exception(mock_bot):
    """Test run_cycle handles sell execution exception."""
    mock_bot.collector.collect.return_value = {
        "price": 55000000.0,
        "rsi": 75.0,
        "bb_position": "upper_half",
        "price_changes": {"5": 0.05, "10": 0.10, "20": 0.15},
        "fear_greed": 70,
    }
    mock_bot.exchange.get_position_amount.side_effect = [0.001, 0.001]
    mock_bot.strategy.determine_action.return_value = {
        "type": "sell",
        "amount": 0,
        "reason": "RSI overbought",
    }
    mock_bot.exchange.get_avg_buy_price.return_value = 50000000.0
    mock_bot.risk_manager.should_stop_loss.return_value = False
    mock_bot.risk_manager.should_take_profit.return_value = (False, 0)
    mock_bot.exchange.sell_market_order.side_effect = Exception("Sell failed")

    mock_bot.run_cycle("KRW-BTC", 1.0)

    mock_bot.notifier.send_sync.assert_called_once()
    mock_bot.notifier.format_error_message.assert_called_once()
