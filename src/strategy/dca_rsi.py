from dataclasses import dataclass, field

from loguru import logger


@dataclass
class DcaRsiStrategy:
    base_amount: float = 5000.0
    rsi_buy_threshold: float = 30.0
    rsi_sell_threshold: float = 70.0
    buy_multipliers: dict = field(
        default_factory=lambda: {"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )

    def determine_action(self, rsi: float, has_position: bool = False) -> dict:
        """Determine trading action based on RSI value.

        Returns dict with keys: type (buy|sell|skip), amount (for buy), reason
        """
        if rsi < self.rsi_buy_threshold:
            multiplier = self.buy_multipliers.get("oversold", 1.5)
            amount = self.base_amount * multiplier
            return {
                "type": "buy",
                "amount": amount,
                "reason": f"RSI oversold ({rsi:.1f} < {self.rsi_buy_threshold})",
            }

        elif rsi < 45:
            multiplier = self.buy_multipliers.get("low", 1.2)
            amount = self.base_amount * multiplier
            return {"type": "buy", "amount": amount, "reason": f"RSI low ({rsi:.1f})"}

        elif rsi < 55:
            multiplier = self.buy_multipliers.get("neutral", 1.0)
            amount = self.base_amount * multiplier
            return {"type": "buy", "amount": amount, "reason": f"RSI neutral ({rsi:.1f})"}

        elif rsi < self.rsi_sell_threshold:
            return {"type": "skip", "amount": 0, "reason": f"RSI elevated ({rsi:.1f}), skipping"}

        else:
            if has_position:
                return {
                    "type": "sell",
                    "amount": 0,
                    "reason": f"RSI overbought ({rsi:.1f} > {self.rsi_sell_threshold})",
                }
            return {
                "type": "skip",
                "amount": 0,
                "reason": f"RSI overbought ({rsi:.1f}) but no position",
            }

    def apply_overrides(self, overrides: dict) -> None:
        """Apply parameter overrides from LLM or RL layer."""
        if "rsi_buy_threshold" in overrides:
            self.rsi_buy_threshold = float(overrides["rsi_buy_threshold"])
        if "rsi_sell_threshold" in overrides:
            self.rsi_sell_threshold = float(overrides["rsi_sell_threshold"])
        if "buy_multiplier" in overrides:
            # Apply single multiplier to all tiers proportionally
            base = overrides["buy_multiplier"]
            self.buy_multipliers["oversold"] = base * 1.25
            self.buy_multipliers["low"] = base
            self.buy_multipliers["neutral"] = base * 0.83
        if "base_amount" in overrides:
            self.base_amount = float(overrides["base_amount"])
        logger.debug(f"Strategy overrides applied: {overrides}")
