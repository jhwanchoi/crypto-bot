import pytest

from src.strategy.dca_rsi import DcaRsiStrategy


class TestDcaRsiStrategy:
    def setup_method(self):
        """Setup test fixtures."""
        self.strategy = DcaRsiStrategy(
            base_amount=5000.0,
            rsi_buy_threshold=30.0,
            rsi_sell_threshold=70.0,
            buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0},
        )

    def test_determine_action_oversold(self):
        """Test action when RSI is oversold (< 30)."""
        action = self.strategy.determine_action(rsi=25.0, has_position=False)
        assert action["type"] == "buy"
        assert action["amount"] == 5000.0 * 1.5  # 7500.0
        assert "oversold" in action["reason"].lower()
        assert "25.0" in action["reason"]

    def test_determine_action_low_rsi(self):
        """Test action when RSI is low (30-45)."""
        action = self.strategy.determine_action(rsi=35.0, has_position=False)
        assert action["type"] == "buy"
        assert action["amount"] == 5000.0 * 1.2  # 6000.0
        assert "low" in action["reason"].lower()
        assert "35.0" in action["reason"]

    def test_determine_action_neutral_rsi(self):
        """Test action when RSI is neutral (45-55)."""
        action = self.strategy.determine_action(rsi=50.0, has_position=False)
        assert action["type"] == "buy"
        assert action["amount"] == 5000.0 * 1.0  # 5000.0
        assert "neutral" in action["reason"].lower()
        assert "50.0" in action["reason"]

    def test_determine_action_elevated_rsi(self):
        """Test action when RSI is elevated (55-70) - should skip."""
        action = self.strategy.determine_action(rsi=60.0, has_position=False)
        assert action["type"] == "skip"
        assert action["amount"] == 0
        assert "elevated" in action["reason"].lower()
        assert "60.0" in action["reason"]

    def test_determine_action_overbought_with_position(self):
        """Test action when RSI is overbought (>70) and has position - should sell."""
        action = self.strategy.determine_action(rsi=75.0, has_position=True)
        assert action["type"] == "sell"
        assert action["amount"] == 0
        assert "overbought" in action["reason"].lower()
        assert "75.0" in action["reason"]

    def test_determine_action_overbought_without_position(self):
        """Test action when RSI is overbought (>70) but no position - should skip."""
        action = self.strategy.determine_action(rsi=75.0, has_position=False)
        assert action["type"] == "skip"
        assert action["amount"] == 0
        assert "overbought" in action["reason"].lower()
        assert "no position" in action["reason"].lower()

    def test_determine_action_boundary_buy_threshold(self):
        """Test action at exact buy threshold boundary."""
        action = self.strategy.determine_action(rsi=30.0, has_position=False)
        assert action["type"] == "buy"
        assert action["amount"] == 5000.0 * 1.2  # Should use "low" multiplier

    def test_determine_action_boundary_sell_threshold(self):
        """Test action at exact sell threshold boundary (70.0 is overbought, not elevated)."""
        # rsi=70.0 with threshold=70.0: 70 < 70 is False → falls to overbought
        action = self.strategy.determine_action(rsi=70.0, has_position=False)
        assert action["type"] == "skip"
        assert "overbought" in action["reason"].lower()

    def test_apply_overrides_rsi_thresholds(self):
        """Test applying overrides to RSI thresholds."""
        overrides = {
            "rsi_buy_threshold": 35.0,
            "rsi_sell_threshold": 65.0,
        }
        self.strategy.apply_overrides(overrides)
        assert self.strategy.rsi_buy_threshold == 35.0
        assert self.strategy.rsi_sell_threshold == 65.0

    def test_apply_overrides_buy_multiplier(self):
        """Test applying buy multiplier override scales all tiers proportionally."""
        overrides = {"buy_multiplier": 2.0}
        self.strategy.apply_overrides(overrides)

        # Should scale proportionally: base * 2.0
        assert self.strategy.buy_multipliers["oversold"] == pytest.approx(2.0 * 1.25, rel=1e-6)
        assert self.strategy.buy_multipliers["low"] == pytest.approx(2.0, rel=1e-6)
        assert self.strategy.buy_multipliers["neutral"] == pytest.approx(2.0 * 0.83, rel=1e-6)

    def test_apply_overrides_base_amount(self):
        """Test applying base amount override."""
        overrides = {"base_amount": 10_000.0}
        self.strategy.apply_overrides(overrides)
        assert self.strategy.base_amount == 10_000.0

    def test_apply_overrides_multiple_parameters(self):
        """Test applying multiple overrides simultaneously."""
        overrides = {
            "rsi_buy_threshold": 25.0,
            "rsi_sell_threshold": 75.0,
            "buy_multiplier": 1.5,
            "base_amount": 8_000.0,
        }
        self.strategy.apply_overrides(overrides)

        assert self.strategy.rsi_buy_threshold == 25.0
        assert self.strategy.rsi_sell_threshold == 75.0
        assert self.strategy.base_amount == 8_000.0
        assert self.strategy.buy_multipliers["low"] == pytest.approx(1.5, rel=1e-6)

    def test_apply_overrides_empty_dict(self):
        """Test applying empty overrides doesn't change anything."""
        original_buy_threshold = self.strategy.rsi_buy_threshold
        original_sell_threshold = self.strategy.rsi_sell_threshold
        original_base_amount = self.strategy.base_amount

        self.strategy.apply_overrides({})

        assert self.strategy.rsi_buy_threshold == original_buy_threshold
        assert self.strategy.rsi_sell_threshold == original_sell_threshold
        assert self.strategy.base_amount == original_base_amount

    def test_determine_action_with_modified_thresholds(self):
        """Test that actions change correctly after threshold overrides."""
        # Original: buy threshold = 30
        action_before = self.strategy.determine_action(rsi=32.0, has_position=False)
        assert action_before["type"] == "buy"

        # After override: buy threshold = 35
        self.strategy.apply_overrides({"rsi_buy_threshold": 35.0})
        action_after = self.strategy.determine_action(rsi=32.0, has_position=False)
        assert action_after["type"] == "buy"  # Still buy, but now in "oversold" category
        assert action_after["amount"] == 5000.0 * 1.5  # Uses oversold multiplier
