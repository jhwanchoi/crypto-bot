import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

from src.data.collector import MarketDataCollector
from src.exchange.upbit_client import UpbitClient
from src.llm.advisor import LLMAdvisor
from src.notification.telegram import TelegramNotifier
from src.risk.manager import RiskManager
from src.rl.finrl_agent import FinRLAgent
from src.strategy.dca_rsi import DcaRsiStrategy
from src.utils.config import load_config
from src.utils.db import TradeDB
from src.utils.logger import setup_logger


class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)

        # Setup logger
        log_cfg = self.config.get("logging", {})
        setup_logger(
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("file", "logs/crypto_bot.log"),
            rotation=log_cfg.get("rotation", "10 MB"),
        )
        logger.info("=== Trading Bot Initializing ===")

        # Initialize components
        self.paper_mode = self.config.get("mode", "paper") == "paper"
        logger.info(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")

        # Exchange
        exchange_cfg = self.config["exchange"]
        self.exchange = UpbitClient(
            access_key=exchange_cfg.get("access_key", ""),
            secret_key=exchange_cfg.get("secret_key", ""),
            paper_mode=self.paper_mode,
        )

        # Data collector
        self.collector = MarketDataCollector(self.exchange)

        # Strategy
        strategy_cfg = self.config["strategy"]
        self.strategy = DcaRsiStrategy(
            base_amount=strategy_cfg.get("base_amount_krw", 5000),
            rsi_buy_threshold=strategy_cfg.get("rsi_buy_threshold", 30),
            rsi_sell_threshold=strategy_cfg.get("rsi_sell_threshold", 70),
            buy_multipliers=strategy_cfg.get(
                "buy_multipliers", {"oversold": 1.5, "low": 1.2, "neutral": 1.0}
            ),
        )
        self.rsi_period = strategy_cfg.get("rsi_period", 14)

        # Risk manager
        risk_cfg = self.config.get("risk", {})
        self.risk_manager = RiskManager(**risk_cfg)

        # RL agent
        rl_cfg = self.config.get("rl", {})
        self.rl_enabled = rl_cfg.get("enabled", False)
        self.rl_agent = None
        if self.rl_enabled:
            try:
                self.rl_agent = FinRLAgent(
                    model_path=rl_cfg.get("model_path", "models/finrl_pretrained.zip"),
                    fallback=rl_cfg.get("fallback_to_default", True),
                )
                logger.info(
                    f"RL Agent: {'available' if self.rl_agent.is_available else 'fallback mode'}"
                )
            except Exception as e:
                logger.error(f"RL agent init failed: {e}")
                self.rl_agent = None

        # LLM advisor
        llm_cfg = self.config.get("llm", {})
        self.llm_enabled = llm_cfg.get("enabled", False)
        self.llm_advisor = None
        if self.llm_enabled:
            try:
                self.llm_advisor = LLMAdvisor(
                    model_id=llm_cfg.get("model_id", "anthropic.claude-opus-4-6-v1"),
                    region=llm_cfg.get("region", "us-east-1"),
                    enabled=True,
                    temperature=llm_cfg.get("temperature", 0),
                    max_tokens=llm_cfg.get("max_tokens", 1024),
                    profile_name=llm_cfg.get("aws_profile"),
                    verify_ssl=llm_cfg.get("verify_ssl", True),
                )
                logger.info("LLM Advisor: initialized")
            except Exception as e:
                logger.error(f"LLM advisor init failed: {e}")
                self.llm_advisor = None

        # Notification
        notif_cfg = self.config.get("notification", {}).get("telegram", {})
        self.notifier = TelegramNotifier(
            bot_token=notif_cfg.get("bot_token", ""),
            chat_id=notif_cfg.get("chat_id", ""),
            enabled=notif_cfg.get("enabled", False),
        )

        # Database
        self.db = TradeDB(db_path="data/trades.db")

        # Assets configuration
        self.assets = self.config.get("assets", [])
        self.interval_hours = strategy_cfg.get("interval_hours", 4)

        logger.info(f"Assets configured: {[a['ticker'] for a in self.assets]}")
        logger.info("=== Trading Bot Ready ===")

    def run_cycle(self, ticker: str, allocation: float) -> None:
        """Execute a single trading cycle for one ticker."""
        logger.info(f"--- Running cycle for {ticker} (allocation={allocation}) ---")

        try:
            # 1. Collect market data
            data = self.collector.collect(ticker, rsi_period=self.rsi_period)
            if data is None:
                logger.error(f"Failed to collect data for {ticker}")
                self.notifier.send_sync(
                    self.notifier.format_error_message(f"Data collection failed for {ticker}")
                )
                return

            price = data["price"]
            rsi = data["rsi"]
            bb_position = data["bb_position"]
            price_changes = data["price_changes"]
            fear_greed = data["fear_greed"]

            logger.info(
                f"{ticker}: price={price:.2f}, RSI={rsi:.1f}, BB={bb_position}, F&G={fear_greed}"
            )

            # 2. Get RL signal (if enabled)
            rl_signal = None
            if self.rl_enabled and self.rl_agent and self.rl_agent.is_available:
                try:
                    # Build state vector for RL agent
                    state = np.array(
                        [
                            rsi / 100.0,
                            price_changes.get("5", 0),
                            price_changes.get("20", 0),
                            fear_greed / 100.0,
                        ],
                        dtype=np.float32,
                    )
                    rl_result = self.rl_agent.predict(state)
                    rl_signal = {
                        "action": rl_result.action,
                        "confidence": rl_result.confidence,
                        "params": rl_result.suggested_params,
                    }
                    logger.debug(f"RL signal: {rl_signal}")
                except Exception as e:
                    logger.error(f"RL prediction failed: {e}")
                    rl_signal = None

            # 3. Get LLM analysis (if enabled)
            llm_analysis = None
            if self.llm_enabled and self.llm_advisor:
                try:
                    llm_analysis = self.llm_advisor.analyze(
                        ticker=ticker,
                        rsi=rsi,
                        price=price,
                        price_changes=price_changes,
                        bb_position=bb_position,
                        rl_signal=rl_signal or {},
                        fear_greed=fear_greed,
                    )
                    if llm_analysis:
                        logger.info(
                            f"LLM analysis: {llm_analysis.get('market_phase', 'unknown')}, reasoning: {llm_analysis.get('reasoning', 'N/A')[:50]}"
                        )
                except Exception as e:
                    logger.error(f"LLM analysis failed: {e}")
                    llm_analysis = None

            # 4. Apply overrides to strategy
            if llm_analysis:
                overrides = {
                    "rsi_buy_threshold": llm_analysis.get("rsi_buy_threshold"),
                    "rsi_sell_threshold": llm_analysis.get("rsi_sell_threshold"),
                    "buy_multiplier": llm_analysis.get("buy_multiplier"),
                }
                self.strategy.apply_overrides(overrides)
            elif rl_signal and rl_signal.get("params"):
                self.strategy.apply_overrides(rl_signal["params"])

            # 5. Check for action override
            if llm_analysis and llm_analysis.get("action_override") == "skip":
                logger.info(f"LLM override: skip trading for {ticker}")
                return
            if llm_analysis and llm_analysis.get("action_override") == "sell_all":
                logger.warning(f"LLM override: sell all {ticker}")
                # Force sell logic could be implemented here
                return

            # 6. Determine action via DCA+RSI strategy
            currency = ticker.split("-")[1]
            has_position = self.exchange.get_position_amount(currency) > 0
            action = self.strategy.determine_action(rsi, has_position=has_position)

            logger.info(f"Strategy decision: {action}")

            # 7. Risk check and execute
            if action["type"] == "buy":
                current_cash = self.exchange.get_balance("KRW")
                daily_count = self.db.get_daily_trade_count()

                # Calculate seed dynamically for paper mode
                total_seed = current_cash + sum(
                    self.exchange.get_position_amount(a["ticker"].split("-")[1])
                    * self.exchange.get_current_price(a["ticker"])
                    for a in self.assets
                )

                allowed, reason = self.risk_manager.check_buy_allowed(
                    total_seed=total_seed,
                    current_cash=current_cash,
                    buy_amount=action["amount"],
                    daily_trade_count=daily_count,
                )

                if not allowed:
                    logger.warning(f"Buy blocked by risk manager: {reason}")
                    return

                # Execute buy
                try:
                    result = self.exchange.buy_market_order(ticker, action["amount"])
                    if "error" not in result:
                        logger.info(f"Buy executed: {result}")
                        self.db.insert_trade(
                            ticker=ticker,
                            side="buy",
                            price=price,
                            amount=result.get("amount", 0),
                            total_krw=action["amount"],
                            rsi=rsi,
                            rl_signal=str(rl_signal) if rl_signal else None,
                            llm_reasoning=llm_analysis.get("reasoning") if llm_analysis else None,
                        )
                        # Send notification
                        msg = self.notifier.format_trade_message(
                            ticker=ticker,
                            side="buy",
                            price=price,
                            amount=result.get("amount", 0),
                            total_krw=action["amount"],
                            rsi=rsi,
                            reasoning=action.get("reason", ""),
                        )
                        self.notifier.send_sync(msg)
                    else:
                        logger.error(f"Buy order failed: {result}")
                except Exception as e:
                    logger.error(f"Buy execution error: {e}")
                    self.notifier.send_sync(self.notifier.format_error_message(str(e)))

            elif action["type"] == "sell":
                # Check stop-loss / take-profit
                avg_price = self.exchange.get_avg_buy_price(currency)
                position_amount = self.exchange.get_position_amount(currency)

                if position_amount <= 0:
                    logger.info(f"No position to sell for {ticker}")
                    return

                sell_amount = position_amount
                sell_reason = action.get("reason", "")

                # Stop-loss check
                if self.risk_manager.should_stop_loss(avg_price, price):
                    sell_reason = f"Stop-loss triggered: {sell_reason}"
                    logger.warning(sell_reason)

                # Take-profit check
                should_tp, tp_ratio = self.risk_manager.should_take_profit(avg_price, price)
                if should_tp:
                    sell_amount = position_amount * tp_ratio
                    sell_reason = f"Take-profit (partial {tp_ratio * 100:.0f}%): {sell_reason}"
                    logger.info(sell_reason)

                # Execute sell
                try:
                    result = self.exchange.sell_market_order(ticker, sell_amount)
                    if "error" not in result:
                        logger.info(f"Sell executed: {result}")
                        total_krw = sell_amount * price
                        self.db.insert_trade(
                            ticker=ticker,
                            side="sell",
                            price=price,
                            amount=sell_amount,
                            total_krw=total_krw,
                            rsi=rsi,
                            rl_signal=str(rl_signal) if rl_signal else None,
                            llm_reasoning=llm_analysis.get("reasoning") if llm_analysis else None,
                        )
                        # Send notification
                        msg = self.notifier.format_trade_message(
                            ticker=ticker,
                            side="sell",
                            price=price,
                            amount=sell_amount,
                            total_krw=total_krw,
                            rsi=rsi,
                            reasoning=sell_reason,
                        )
                        self.notifier.send_sync(msg)
                    else:
                        logger.error(f"Sell order failed: {result}")
                except Exception as e:
                    logger.error(f"Sell execution error: {e}")
                    self.notifier.send_sync(self.notifier.format_error_message(str(e)))

            else:
                logger.info(f"No action: {action.get('reason', '')}")

        except Exception as e:
            logger.error(f"Cycle error for {ticker}: {e}")
            self.notifier.send_sync(self.notifier.format_error_message(f"Cycle error: {e}"))

    def run(self) -> None:
        """Run trading cycles for all configured assets."""
        logger.info("=== Starting trading run ===")
        for asset in self.assets:
            ticker = asset["ticker"]
            allocation = asset.get("allocation", 1.0)
            self.run_cycle(ticker, allocation)
        logger.info("=== Trading run completed ===")

    def start(self) -> None:
        """Start the bot with scheduler."""
        logger.info(f"Starting scheduler with interval={self.interval_hours}h")
        scheduler = BlockingScheduler()
        scheduler.add_job(self.run, "interval", hours=self.interval_hours)

        # Run immediately on start
        self.run()

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down trading bot")


if __name__ == "__main__":
    bot = TradingBot(config_path="config/settings.yaml")
    bot.start()
