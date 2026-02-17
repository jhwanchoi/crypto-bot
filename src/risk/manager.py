from dataclasses import dataclass

from loguru import logger


@dataclass
class RiskConfig:
    max_buy_per_trade_pct: float = 5.0
    max_daily_trades: int = 6
    stop_loss_pct: float = 8.0
    take_profit_pct: float = 15.0
    take_profit_sell_ratio: float = 0.3
    max_position_pct: float = 80.0
    min_cash_pct: float = 20.0


class RiskManager:
    def __init__(self, **kwargs):
        self.config = RiskConfig(**kwargs)

    def check_buy_allowed(
        self, total_seed: float, current_cash: float, buy_amount: float, daily_trade_count: int
    ) -> tuple[bool, str]:
        """Check if a buy order is allowed by risk rules. Returns (allowed, reason)."""
        # Check daily trade limit
        if daily_trade_count >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        # Check max buy per trade
        max_amount = total_seed * (self.config.max_buy_per_trade_pct / 100)
        if buy_amount > max_amount:
            return False, f"Buy amount {buy_amount} exceeds max per trade {max_amount}"

        # Check minimum cash reserve
        cash_after = current_cash - buy_amount
        min_cash = total_seed * (self.config.min_cash_pct / 100)
        if cash_after < min_cash:
            return False, f"Cash after buy ({cash_after:.0f}) below minimum ({min_cash:.0f})"

        # Check max position ratio
        position_value = total_seed - current_cash + buy_amount
        max_position = total_seed * (self.config.max_position_pct / 100)
        if position_value > max_position:
            return False, f"Position value ({position_value:.0f}) exceeds max ({max_position:.0f})"

        return True, "OK"

    def should_stop_loss(self, avg_price: float, current_price: float) -> bool:
        """Check if stop-loss should trigger."""
        if avg_price <= 0:
            return False
        loss_pct = ((avg_price - current_price) / avg_price) * 100
        triggered = loss_pct >= self.config.stop_loss_pct
        if triggered:
            logger.warning(
                f"STOP LOSS triggered: loss={loss_pct:.1f}% >= {self.config.stop_loss_pct}%"
            )
        return triggered

    def should_take_profit(self, avg_price: float, current_price: float) -> tuple[bool, float]:
        """Check if take-profit should trigger. Returns (should_sell, sell_ratio)."""
        if avg_price <= 0:
            return False, 0.0
        gain_pct = ((current_price - avg_price) / avg_price) * 100
        if gain_pct >= self.config.take_profit_pct:
            logger.info(
                f"TAKE PROFIT triggered: gain={gain_pct:.1f}% >= {self.config.take_profit_pct}%"
            )
            return True, self.config.take_profit_sell_ratio
        return False, 0.0

    def calculate_safe_buy_amount(
        self, base_amount: float, total_seed: float, current_cash: float, daily_trade_count: int
    ) -> float:
        """Calculate the maximum safe buy amount respecting all risk rules."""
        max_per_trade = total_seed * (self.config.max_buy_per_trade_pct / 100)
        min_cash = total_seed * (self.config.min_cash_pct / 100)
        max_by_cash = current_cash - min_cash

        safe_amount = min(base_amount, max_per_trade, max(0, max_by_cash))
        return max(0, safe_amount)
