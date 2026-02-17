from src.llm.advisor import DEFAULT_RESPONSE, LLMAdvisor
from src.llm.prompts import build_analysis_prompt


class TestBuildAnalysisPrompt:
    def test_prompt_contains_expected_fields(self):
        prompt = build_analysis_prompt(
            ticker="BTC-USD",
            rsi=35.0,
            price=50000.0,
            price_change_5=2.5,
            price_change_20=-3.2,
            bb_position="middle",
            rl_signal={"action": "buy", "confidence": 0.8},
            fear_greed_index=65,
        )

        assert "BTC-USD" in prompt
        assert "35.0" in prompt or "35.00" in prompt
        assert "50000" in prompt
        assert "2.5" in prompt or "2.50" in prompt
        assert "-3.2" in prompt or "-3.20" in prompt
        assert "middle" in prompt
        assert "buy" in prompt
        assert "65" in prompt
        assert "Fear & Greed Index" in prompt or "fear_greed_index" in prompt


class TestLLMAdvisor:
    def test_parse_valid_json(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        raw = """{
            "market_phase": "bullish",
            "rsi_buy_threshold": 28,
            "rsi_sell_threshold": 72,
            "buy_multiplier": 1.5,
            "take_profit_pct": 18,
            "action_override": null,
            "reasoning": "Test reasoning",
            "risk_alert": null
        }"""
        result = advisor.parse_response(raw)
        assert result["market_phase"] == "bullish"
        assert result["rsi_buy_threshold"] == 28
        assert result["rsi_sell_threshold"] == 72
        assert result["buy_multiplier"] == 1.5
        assert result["take_profit_pct"] == 18

    def test_parse_markdown_wrapped_json(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        raw = """```json
{
    "market_phase": "bearish",
    "rsi_buy_threshold": 32,
    "rsi_sell_threshold": 68,
    "buy_multiplier": 0.9,
    "take_profit_pct": 12,
    "action_override": "skip",
    "reasoning": "High volatility",
    "risk_alert": "Market uncertainty"
}
```"""
        result = advisor.parse_response(raw)
        assert result["market_phase"] == "bearish"
        assert result["rsi_buy_threshold"] == 32
        assert result["action_override"] == "skip"
        assert result["risk_alert"] == "Market uncertainty"

    def test_parse_clamps_values(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        raw = """{
            "market_phase": "bullish",
            "rsi_buy_threshold": 10,
            "rsi_sell_threshold": 95,
            "buy_multiplier": 5.0,
            "take_profit_pct": 50,
            "action_override": null,
            "reasoning": "Test",
            "risk_alert": null
        }"""
        result = advisor.parse_response(raw)
        assert result["rsi_buy_threshold"] == 20  # clamped from 10
        assert result["rsi_sell_threshold"] == 85  # clamped from 95
        assert result["buy_multiplier"] == 2.5  # clamped from 5.0
        assert result["take_profit_pct"] == 30  # clamped from 50

    def test_parse_invalid_json_returns_defaults(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        raw = "This is not valid JSON"
        result = advisor.parse_response(raw)
        assert result == DEFAULT_RESPONSE

    def test_advisor_disabled_returns_none(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        result = advisor.analyze(
            ticker="BTC-USD",
            rsi=35.0,
            price=50000.0,
            price_changes={"5": 2.5, "20": -3.2},
            bb_position="middle",
            rl_signal={"action": "buy"},
            fear_greed=65,
        )
        assert result is None

    def test_get_default_params(self):
        advisor = LLMAdvisor(model_id="test-model", enabled=False)
        defaults = advisor.get_default_params()
        assert defaults["market_phase"] == "unknown"
        assert defaults["rsi_buy_threshold"] == 30
        assert defaults["rsi_sell_threshold"] == 70
        assert defaults["buy_multiplier"] == 1.0
        assert defaults["take_profit_pct"] == 15

    def test_analyze_with_successful_bedrock_response(self):
        """Test analyze method with successful bedrock response."""
        from unittest.mock import MagicMock, patch

        mock_bedrock = MagicMock()
        mock_bedrock.invoke.return_value = """{
            "market_phase": "bullish",
            "rsi_buy_threshold": 28,
            "rsi_sell_threshold": 72,
            "buy_multiplier": 1.5,
            "take_profit_pct": 18,
            "action_override": null,
            "reasoning": "Strong uptrend",
            "risk_alert": null
        }"""

        with patch("src.llm.advisor.BedrockClient", return_value=mock_bedrock):
            advisor = LLMAdvisor(model_id="test-model", enabled=True)
            result = advisor.analyze(
                ticker="BTC-USD",
                rsi=35.0,
                price=50000.0,
                price_changes={"5": 2.5, "20": -3.2},
                bb_position="middle",
                rl_signal={"action": "buy"},
                fear_greed=65,
            )

            assert result is not None
            assert result["market_phase"] == "bullish"
            assert result["rsi_buy_threshold"] == 28
            assert result["buy_multiplier"] == 1.5
            mock_bedrock.invoke.assert_called_once()

    def test_analyze_when_bedrock_returns_none(self):
        """Test analyze method when bedrock returns None."""
        from unittest.mock import MagicMock, patch

        mock_bedrock = MagicMock()
        mock_bedrock.invoke.return_value = None

        with patch("src.llm.advisor.BedrockClient", return_value=mock_bedrock):
            advisor = LLMAdvisor(model_id="test-model", enabled=True)
            result = advisor.analyze(
                ticker="BTC-USD",
                rsi=35.0,
                price=50000.0,
                price_changes={"5": 2.5, "20": -3.2},
                bb_position="middle",
                rl_signal={"action": "buy"},
                fear_greed=65,
            )

            assert result is None
            mock_bedrock.invoke.assert_called_once()

    def test_bedrock_client_initialization(self):
        """Test BedrockClient is initialized when advisor is enabled."""
        from unittest.mock import patch

        with patch("src.llm.advisor.BedrockClient") as mock_bedrock_class:
            advisor = LLMAdvisor(
                model_id="test-model",
                region="us-west-2",
                enabled=True,
                temperature=0.5,
                max_tokens=2048,
                profile_name="test-profile",
                verify_ssl=False,
            )

            assert advisor.bedrock is not None
            mock_bedrock_class.assert_called_once_with(
                model_id="test-model",
                region="us-west-2",
                temperature=0.5,
                max_tokens=2048,
                profile_name="test-profile",
                verify_ssl=False,
            )
