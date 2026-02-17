import pandas as pd
from loguru import logger

from src.data.sentiment import get_fear_greed_index
from src.exchange.upbit_client import UpbitClient
from src.strategy.indicators import (
    calculate_bollinger_bands,
    calculate_price_change,
    calculate_rsi,
    get_bb_position,
)


class MarketDataCollector:
    """Collects and aggregates market data for a ticker."""

    def __init__(self, exchange: UpbitClient):
        self.exchange = exchange

    def collect(self, ticker: str, rsi_period: int = 14) -> dict | None:
        """Collect all market data needed for decision making.

        Returns dict with: price, ohlcv, rsi, bb_position, price_changes, fear_greed
        """
        try:
            # Need enough candles for indicators
            count = max(rsi_period + 5, 25)
            ohlcv = self.exchange.get_ohlcv(ticker, interval="minute240", count=count)
            if ohlcv is None or len(ohlcv) < rsi_period:
                logger.error(f"Insufficient OHLCV data for {ticker}")
                return None

            closes = ohlcv["close"]
            current_price = float(closes.iloc[-1])

            # RSI
            rsi_series = calculate_rsi(closes, period=rsi_period)
            rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

            # Bollinger Bands
            upper, _middle, lower = calculate_bollinger_bands(closes)
            bb_pos = get_bb_position(
                current_price,
                float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else current_price,
                float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else current_price,
            )

            # Price changes
            price_changes = {
                "5": calculate_price_change(closes, 5),
                "10": calculate_price_change(closes, 10),
                "20": calculate_price_change(closes, 20),
            }

            # Fear & Greed
            fear_greed = get_fear_greed_index()

            return {
                "ticker": ticker,
                "price": current_price,
                "rsi": rsi,
                "bb_position": bb_pos,
                "price_changes": price_changes,
                "fear_greed": fear_greed,
                "ohlcv": ohlcv,
            }
        except Exception as e:
            logger.error(f"Data collection failed for {ticker}: {e}")
            return None
