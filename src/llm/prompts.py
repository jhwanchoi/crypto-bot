def build_analysis_prompt(
    ticker: str,
    rsi: float,
    price: float,
    price_change_5: float,
    price_change_20: float,
    bb_position: str,
    rl_signal: dict,
    fear_greed_index: int,
) -> str:
    """Build analysis prompt for Claude."""
    return f"""You are a crypto trading advisor analyzing market conditions for {ticker}.

Current Market Data:
- Price: ${price:.2f}
- RSI: {rsi:.2f}
- Price change (5-period): {price_change_5:.2f}%
- Price change (20-period): {price_change_20:.2f}%
- Bollinger Band Position: {bb_position}
- RL Agent Signal: {rl_signal}
- Fear & Greed Index: {fear_greed_index}

Based on this data, provide your analysis and parameter recommendations.

Output ONLY a JSON object with this exact schema (no markdown, no explanation):
{{
  "market_phase": "bullish|bearish|sideways",
  "rsi_buy_threshold": 25-35,
  "rsi_sell_threshold": 65-80,
  "buy_multiplier": 0.8-2.0,
  "take_profit_pct": 10-25,
  "action_override": null or "skip" or "sell_all",
  "reasoning": "brief explanation",
  "risk_alert": null or "warning message"
}}"""
