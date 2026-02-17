import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.indicators import calculate_rsi


class Backtester:
    """Simple backtester that simulates DCA+RSI strategy on historical data."""

    def __init__(
        self,
        initial_cash: float = 500000.0,
        base_amount: float = 5000.0,
        rsi_period: int = 14,
        rsi_buy_threshold: float = 30.0,
        rsi_sell_threshold: float = 70.0,
    ):
        self.initial_cash = initial_cash
        self.base_amount = base_amount
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    def run(self, prices: pd.Series) -> dict:
        """Simulate trading on historical prices.

        Returns dict with: total_return_pct, total_trades, win_rate, sharpe_ratio,
                          max_drawdown, min_cash, final_value
        """
        if len(prices) < self.rsi_period + 1:
            logger.warning("Insufficient price data for backtest")
            return {
                "total_return_pct": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "min_cash": self.initial_cash,
                "final_value": self.initial_cash,
            }

        # Calculate RSI
        rsi = calculate_rsi(prices, period=self.rsi_period)

        # Initialize tracking
        cash = self.initial_cash
        position = 0.0
        avg_price = 0.0
        trades = []
        equity_curve = []
        min_cash = cash

        for i in range(self.rsi_period, len(prices)):
            price = prices.iloc[i]
            current_rsi = rsi.iloc[i]

            if pd.isna(current_rsi):
                continue

            # Calculate current portfolio value
            portfolio_value = cash + position * price
            equity_curve.append(portfolio_value)

            # Trading logic
            if current_rsi < self.rsi_buy_threshold and cash >= self.base_amount:
                # Buy signal
                buy_amount = min(self.base_amount, cash)
                coins_bought = buy_amount / price
                cash -= buy_amount
                min_cash = min(min_cash, cash)

                # Update position and average price
                if position > 0:
                    avg_price = (avg_price * position + price * coins_bought) / (
                        position + coins_bought
                    )
                else:
                    avg_price = price
                position += coins_bought

                trades.append(
                    {
                        "index": i,
                        "side": "buy",
                        "price": price,
                        "amount": coins_bought,
                        "rsi": current_rsi,
                    }
                )
                logger.debug(f"BUY @ {price:.2f}, RSI={current_rsi:.1f}, cash={cash:.0f}")

            elif current_rsi > self.rsi_sell_threshold and position > 0:
                # Sell signal
                sell_value = position * price
                cash += sell_value

                # Calculate P&L
                pnl = sell_value - (avg_price * position)

                trades.append(
                    {
                        "index": i,
                        "side": "sell",
                        "price": price,
                        "amount": position,
                        "rsi": current_rsi,
                        "pnl": pnl,
                    }
                )
                logger.debug(
                    f"SELL @ {price:.2f}, RSI={current_rsi:.1f}, cash={cash:.0f}, P&L={pnl:.0f}"
                )

                position = 0.0
                avg_price = 0.0

        # Final value
        final_value = cash + position * prices.iloc[-1]
        total_return_pct = ((final_value - self.initial_cash) / self.initial_cash) * 100

        # Calculate win rate
        sell_trades = [t for t in trades if t["side"] == "sell"]
        if sell_trades:
            wins = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
            win_rate = (wins / len(sell_trades)) * 100
        else:
            win_rate = 0.0

        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        # Calculate max drawdown
        if len(equity_curve) > 0:
            equity_series = pd.Series(equity_curve)
            cummax = equity_series.cummax()
            drawdown = (equity_series - cummax) / cummax
            max_drawdown = abs(drawdown.min()) * 100
        else:
            max_drawdown = 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "min_cash": round(min_cash, 2),
            "final_value": round(final_value, 2),
        }
