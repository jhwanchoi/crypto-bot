import json

from loguru import logger

from src.llm.bedrock_client import BedrockClient
from src.llm.prompts import build_analysis_prompt

DEFAULT_RESPONSE = {
    "market_phase": "unknown",
    "rsi_buy_threshold": 30,
    "rsi_sell_threshold": 70,
    "buy_multiplier": 1.0,
    "take_profit_pct": 15,
    "action_override": None,
    "reasoning": "Using default parameters (LLM unavailable)",
    "risk_alert": None,
}


class LLMAdvisor:
    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        enabled: bool = True,
        temperature: float = 0,
        max_tokens: int = 1024,
        profile_name: str | None = None,
        verify_ssl: bool = True,
    ):
        self.enabled = enabled
        self.bedrock = None
        if enabled:
            self.bedrock = BedrockClient(
                model_id=model_id,
                region=region,
                temperature=temperature,
                max_tokens=max_tokens,
                profile_name=profile_name,
                verify_ssl=verify_ssl,
            )

    def analyze(
        self,
        ticker: str,
        rsi: float,
        price: float,
        price_changes: dict,
        bb_position: str,
        rl_signal: dict,
        fear_greed: int,
    ) -> dict | None:
        if not self.enabled or self.bedrock is None:
            return None

        prompt = build_analysis_prompt(
            ticker=ticker,
            rsi=rsi,
            price=price,
            price_change_5=price_changes.get("5", 0),
            price_change_20=price_changes.get("20", 0),
            bb_position=bb_position,
            rl_signal=rl_signal,
            fear_greed_index=fear_greed,
        )
        raw = self.bedrock.invoke(prompt)
        if raw is None:
            return None
        return self.parse_response(raw)

    def parse_response(self, raw: str) -> dict:
        try:
            # Extract JSON from response (may have markdown code blocks)
            text = raw.strip()
            if "```" in text:
                start = text.index("```") + 3
                if text[start:].startswith("json"):
                    start += 4
                end = text.index("```", start)
                text = text[start:end].strip()
            result = json.loads(text)
            # Validate and clamp values
            result["rsi_buy_threshold"] = max(20, min(40, result.get("rsi_buy_threshold", 30)))
            result["rsi_sell_threshold"] = max(60, min(85, result.get("rsi_sell_threshold", 70)))
            result["buy_multiplier"] = max(0.5, min(2.5, result.get("buy_multiplier", 1.0)))
            result["take_profit_pct"] = max(5, min(30, result.get("take_profit_pct", 15)))
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return DEFAULT_RESPONSE.copy()

    def get_default_params(self) -> dict:
        return DEFAULT_RESPONSE.copy()
