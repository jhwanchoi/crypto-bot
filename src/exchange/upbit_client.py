import pyupbit
from loguru import logger


class UpbitClient:
    def __init__(self, access_key: str, secret_key: str, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self._upbit = pyupbit.Upbit(access_key, secret_key)
        # Paper trading state
        self._paper_balance = {"KRW": 500000.0}  # default paper seed
        self._paper_positions = {}

    def get_balance(self, ticker: str = "KRW") -> float:
        if self.paper_mode:
            return self._paper_balance.get(ticker, 0.0)
        return self._upbit.get_balance(ticker)

    def get_ohlcv(self, ticker: str, interval: str = "minute240", count: int = 14):
        """Get OHLCV candle data. Works in both paper and live mode (uses real market data)."""
        return pyupbit.get_ohlcv(ticker, interval=interval, count=count)

    def get_current_price(self, ticker: str) -> float:
        return pyupbit.get_current_price(ticker)

    def get_avg_buy_price(self, currency: str) -> float:
        if self.paper_mode:
            pos = self._paper_positions.get(currency, {})
            return pos.get("avg_price", 0.0)
        return self._upbit.get_avg_buy_price(currency)

    def get_position_amount(self, currency: str) -> float:
        if self.paper_mode:
            pos = self._paper_positions.get(currency, {})
            return pos.get("amount", 0.0)
        return self._upbit.get_balance(currency)

    def buy_market_order(self, ticker: str, amount_krw: float) -> dict:
        if self.paper_mode:
            return self._paper_buy(ticker, amount_krw)
        result = self._upbit.buy_market_order(ticker, amount_krw)
        logger.info(f"BUY {ticker}: {amount_krw} KRW -> {result}")
        return result

    def sell_market_order(self, ticker: str, amount: float) -> dict:
        if self.paper_mode:
            return self._paper_sell(ticker, amount)
        result = self._upbit.sell_market_order(ticker, amount)
        logger.info(f"SELL {ticker}: {amount} -> {result}")
        return result

    def _paper_buy(self, ticker: str, amount_krw: float) -> dict:
        currency = ticker.split("-")[1]  # KRW-BTC -> BTC
        price = self.get_current_price(ticker)
        if price is None or price <= 0:
            return {"error": "Cannot get current price"}
        fee = amount_krw * 0.0005  # 0.05% fee
        net_amount_krw = amount_krw - fee
        coin_amount = net_amount_krw / price

        self._paper_balance["KRW"] = self._paper_balance.get("KRW", 0) - amount_krw

        pos = self._paper_positions.get(
            currency, {"amount": 0.0, "avg_price": 0.0, "total_invested": 0.0}
        )
        total_amount = pos["amount"] + coin_amount
        if total_amount > 0:
            pos["avg_price"] = (pos["total_invested"] + net_amount_krw) / total_amount
        pos["amount"] = total_amount
        pos["total_invested"] = pos.get("total_invested", 0) + net_amount_krw
        self._paper_positions[currency] = pos

        logger.info(
            f"[PAPER] BUY {ticker}: {amount_krw} KRW @ {price} = {coin_amount:.8f} {currency}"
        )
        return {"uuid": f"paper-buy-{ticker}", "side": "bid", "price": price, "amount": coin_amount}

    def _paper_sell(self, ticker: str, amount: float) -> dict:
        currency = ticker.split("-")[1]
        price = self.get_current_price(ticker)
        if price is None or price <= 0:
            return {"error": "Cannot get current price"}

        pos = self._paper_positions.get(
            currency, {"amount": 0.0, "avg_price": 0.0, "total_invested": 0.0}
        )
        sell_amount = min(amount, pos["amount"])
        krw_received = sell_amount * price
        fee = krw_received * 0.0005
        net_krw = krw_received - fee

        pos["amount"] -= sell_amount
        if pos["amount"] <= 0:
            pos["amount"] = 0.0
            pos["total_invested"] = 0.0
        else:
            pos["total_invested"] = pos["avg_price"] * pos["amount"]
        self._paper_positions[currency] = pos
        self._paper_balance["KRW"] = self._paper_balance.get("KRW", 0) + net_krw

        logger.info(f"[PAPER] SELL {ticker}: {sell_amount:.8f} @ {price} = {net_krw:.0f} KRW")
        return {
            "uuid": f"paper-sell-{ticker}",
            "side": "ask",
            "price": price,
            "amount": sell_amount,
        }
