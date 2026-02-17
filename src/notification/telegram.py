from loguru import logger


class TelegramNotifier:
    """Telegram notification sender."""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = False):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._bot = None
        if enabled and bot_token and chat_id:
            try:
                import telegram

                self._bot = telegram.Bot(token=bot_token)
            except Exception as e:
                logger.warning(f"Telegram bot init failed: {e}")

    def format_trade_message(
        self,
        ticker: str,
        side: str,
        price: float,
        amount: float,
        total_krw: float,
        rsi: float,
        reasoning: str,
    ) -> str:
        side_kr = "매수" if side == "buy" else "매도"
        return (
            f"{'🟢' if side == 'buy' else '🔴'} {side_kr} 체결\n"
            f"종목: {ticker}\n"
            f"가격: {price:,.0f} KRW\n"
            f"수량: {amount:.8f}\n"
            f"금액: {total_krw:,.0f} KRW\n"
            f"RSI: {rsi:.1f}\n"
            f"사유: {reasoning}"
        )

    def format_error_message(self, error: str) -> str:
        return f"⚠️ 오류 발생\n{error}"

    def format_status_message(self, portfolio: dict) -> str:
        lines = ["📊 포트폴리오 현황"]
        for ticker, info in portfolio.items():
            lines.append(
                f"  {ticker}: {info.get('amount', 0):.8f} (평균가: {info.get('avg_price', 0):,.0f})"
            )
        return "\n".join(lines)

    async def send(self, message: str) -> bool:
        if not self.enabled or self._bot is None:
            logger.debug(f"[Telegram disabled] {message[:50]}...")
            return False
        try:
            await self._bot.send_message(chat_id=self.chat_id, text=message)
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_sync(self, message: str) -> bool:
        """Synchronous send wrapper."""
        if not self.enabled or self._bot is None:
            logger.debug(f"[Telegram disabled] {message[:50]}...")
            return False
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(self.send(message))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Telegram sync send failed: {e}")
            return False
