from src.risk.manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        """Setup test fixtures."""
        self.total_seed = 100_000.0
        self.rm = RiskManager(
            max_buy_per_trade_pct=5.0,
            max_daily_trades=6,
            stop_loss_pct=8.0,
            take_profit_pct=15.0,
            take_profit_sell_ratio=0.3,
            max_position_pct=80.0,
            min_cash_pct=20.0,
        )

    def test_check_buy_allowed_within_limits(self):
        """Test buy is allowed when all constraints are satisfied."""
        allowed, reason = self.rm.check_buy_allowed(
            total_seed=self.total_seed,
            current_cash=50_000.0,
            buy_amount=4_000.0,
            daily_trade_count=3,
        )
        assert allowed is True
        assert reason == "OK"

    def test_check_buy_allowed_daily_limit_exceeded(self):
        """Test buy is rejected when daily trade limit is reached."""
        allowed, reason = self.rm.check_buy_allowed(
            total_seed=self.total_seed,
            current_cash=50_000.0,
            buy_amount=4_000.0,
            daily_trade_count=6,
        )
        assert allowed is False
        assert "Daily trade limit reached" in reason
        assert "6" in reason

    def test_check_buy_allowed_exceeds_max_per_trade(self):
        """Test buy is rejected when amount exceeds max per trade."""
        # Max per trade = 5% of 100k = 5000.0
        allowed, reason = self.rm.check_buy_allowed(
            total_seed=self.total_seed,
            current_cash=50_000.0,
            buy_amount=6_000.0,  # Exceeds 5% limit
            daily_trade_count=2,
        )
        assert allowed is False
        assert "exceeds max per trade" in reason
        assert "6000" in reason

    def test_check_buy_allowed_min_cash_violation(self):
        """Test buy is rejected when it would violate minimum cash reserve."""
        # Min cash = 20% of 100k = 20k
        # Current cash = 22k, buy 3k would leave 19k < 20k
        # buy_amount=3k is under max_per_trade=5k so that check passes
        allowed, reason = self.rm.check_buy_allowed(
            total_seed=self.total_seed,
            current_cash=22_000.0,
            buy_amount=3_000.0,
            daily_trade_count=2,
        )
        assert allowed is False
        assert "below minimum" in reason

    def test_check_buy_allowed_max_position_violation(self):
        """Test buy is rejected when it would exceed max position ratio."""
        # With 80% max_position + 20% min_cash summing to 100%, min_cash
        # always fires before position. Use a separate config with lower
        # min_cash to isolate the position check.
        rm = RiskManager(
            max_buy_per_trade_pct=5.0,
            max_daily_trades=6,
            stop_loss_pct=8.0,
            take_profit_pct=15.0,
            take_profit_sell_ratio=0.3,
            max_position_pct=80.0,
            min_cash_pct=5.0,
        )
        # Min cash = 5% of 100k = 5k
        # cash=15k, buy 4k → cash_after=11k > 5k (passes min_cash)
        # position = 100k - 15k + 4k = 89k > 80k (fails position)
        allowed, reason = rm.check_buy_allowed(
            total_seed=self.total_seed,
            current_cash=15_000.0,
            buy_amount=4_000.0,
            daily_trade_count=2,
        )
        assert allowed is False
        assert "exceeds max" in reason

    def test_should_stop_loss_triggered(self):
        """Test stop-loss triggers at -8% loss."""
        avg_price = 100.0
        current_price = 92.0  # -8% loss
        assert self.rm.should_stop_loss(avg_price, current_price) is True

    def test_should_stop_loss_not_triggered(self):
        """Test stop-loss does not trigger at -5% loss."""
        avg_price = 100.0
        current_price = 95.0  # -5% loss
        assert self.rm.should_stop_loss(avg_price, current_price) is False

    def test_should_stop_loss_zero_avg_price(self):
        """Test stop-loss returns False when avg_price is zero."""
        assert self.rm.should_stop_loss(0.0, 50.0) is False

    def test_should_take_profit_triggered(self):
        """Test take-profit triggers at +15% gain."""
        avg_price = 100.0
        current_price = 115.0  # +15% gain
        should_sell, sell_ratio = self.rm.should_take_profit(avg_price, current_price)
        assert should_sell is True
        assert sell_ratio == 0.3

    def test_should_take_profit_not_triggered(self):
        """Test take-profit does not trigger at +10% gain."""
        avg_price = 100.0
        current_price = 110.0  # +10% gain
        should_sell, sell_ratio = self.rm.should_take_profit(avg_price, current_price)
        assert should_sell is False
        assert sell_ratio == 0.0

    def test_should_take_profit_zero_avg_price(self):
        """Test take-profit returns False when avg_price is zero."""
        should_sell, sell_ratio = self.rm.should_take_profit(0.0, 150.0)
        assert should_sell is False
        assert sell_ratio == 0.0

    def test_calculate_safe_buy_amount_within_all_limits(self):
        """Test safe buy amount when base amount is within all constraints."""
        safe_amount = self.rm.calculate_safe_buy_amount(
            base_amount=4_000.0,
            total_seed=self.total_seed,
            current_cash=50_000.0,
            daily_trade_count=2,
        )
        assert safe_amount == 4_000.0

    def test_calculate_safe_buy_amount_limited_by_max_per_trade(self):
        """Test safe buy amount is capped by max per trade limit."""
        max_per_trade = self.total_seed * 0.05  # 5000.0
        safe_amount = self.rm.calculate_safe_buy_amount(
            base_amount=10_000.0,
            total_seed=self.total_seed,
            current_cash=50_000.0,
            daily_trade_count=2,
        )
        assert safe_amount == max_per_trade

    def test_calculate_safe_buy_amount_limited_by_min_cash(self):
        """Test safe buy amount is capped by minimum cash reserve."""
        # Min cash = 20k, current = 25k, so max buy = 5k
        safe_amount = self.rm.calculate_safe_buy_amount(
            base_amount=10_000.0,
            total_seed=self.total_seed,
            current_cash=25_000.0,
            daily_trade_count=2,
        )
        assert safe_amount == 5_000.0

    def test_calculate_safe_buy_amount_insufficient_cash(self):
        """Test safe buy amount is zero when cash is below minimum."""
        # Min cash = 20k, current = 18k, so no buy allowed
        safe_amount = self.rm.calculate_safe_buy_amount(
            base_amount=5_000.0,
            total_seed=self.total_seed,
            current_cash=18_000.0,
            daily_trade_count=2,
        )
        assert safe_amount == 0.0

    def test_calculate_safe_buy_amount_respects_multiple_constraints(self):
        """Test safe buy amount respects the tightest constraint."""
        # Base = 10k, max_per_trade = 5k, cash allows 3k
        # Should return 3k (tightest constraint)
        safe_amount = self.rm.calculate_safe_buy_amount(
            base_amount=10_000.0,
            total_seed=self.total_seed,
            current_cash=23_000.0,  # 23k - 20k min = 3k available
            daily_trade_count=2,
        )
        assert safe_amount == 3_000.0
