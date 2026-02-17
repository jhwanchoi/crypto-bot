import numpy as np
import pytest

from src.rl.finrl_agent import FinRLAgent
from src.rl.signal import RLSignal, normalize_signal


class TestRLSignal:
    def test_rl_signal_dataclass(self):
        signal = RLSignal(action="buy", confidence=0.8, suggested_params={"foo": "bar"})
        assert signal.action == "buy"
        assert signal.confidence == 0.8
        assert signal.suggested_params == {"foo": "bar"}

    def test_rl_signal_default_params(self):
        signal = RLSignal(action="hold", confidence=0.5)
        assert signal.suggested_params == {}


class TestNormalizeSignal:
    def test_normalize_valid_signal(self):
        raw = {"action": "buy", "confidence": 0.9, "suggested_params": {"test": 1}}
        signal = normalize_signal(raw)
        assert signal.action == "buy"
        assert signal.confidence == 0.9
        assert signal.suggested_params == {"test": 1}

    def test_normalize_clamps_confidence(self):
        raw = {"action": "sell", "confidence": 1.5}
        signal = normalize_signal(raw)
        assert signal.confidence == 1.0

        raw = {"action": "hold", "confidence": -0.5}
        signal = normalize_signal(raw)
        assert signal.confidence == 0.0

    def test_normalize_invalid_action(self):
        raw = {"action": "invalid_action", "confidence": 0.5}
        signal = normalize_signal(raw)
        assert signal.action == "hold"

    def test_normalize_missing_fields(self):
        raw = {}
        signal = normalize_signal(raw)
        assert signal.action == "hold"
        assert signal.confidence == 0.0
        assert signal.suggested_params == {}


class TestFinRLAgent:
    def test_agent_fallback_no_model(self):
        agent = FinRLAgent(model_path="nonexistent_model.zip", fallback=True)
        assert agent.model is None
        assert not agent.is_available

    def test_agent_raises_without_fallback(self):
        with pytest.raises(FileNotFoundError):
            FinRLAgent(model_path="nonexistent_model.zip", fallback=False)

    def test_predict_without_model(self):
        agent = FinRLAgent(model_path="nonexistent_model.zip", fallback=True)
        state = np.array([1.0, 2.0, 3.0])
        signal = agent.predict(state)
        assert signal.action == "hold"
        assert signal.confidence == 0.0
        assert signal.suggested_params == {}

    def test_interpret_action_continuous(self):
        agent = FinRLAgent(model_path="nonexistent_model.zip", fallback=True)

        # Test buy signal
        action = np.array([0.8, 0.1, 0.2, 0.3])
        signal = agent._interpret_action(action)
        assert signal.action == "buy"
        assert signal.confidence > 0
        assert "rsi_buy_threshold" in signal.suggested_params
        assert "buy_multiplier" in signal.suggested_params
        assert "take_profit_pct" in signal.suggested_params

        # Test sell signal
        action = np.array([-0.7, -0.1, -0.2, -0.1])
        signal = agent._interpret_action(action)
        assert signal.action == "sell"
        assert signal.confidence > 0

        # Test hold signal
        action = np.array([0.1, 0.0, 0.0, 0.0])
        signal = agent._interpret_action(action)
        assert signal.action == "hold"

    def test_interpret_action_invalid(self):
        agent = FinRLAgent(model_path="nonexistent_model.zip", fallback=True)

        # Test with insufficient action dimensions
        action = np.array([0.5])
        signal = agent._interpret_action(action)
        assert signal.action == "hold"
        assert signal.confidence == 0.0

        # Test with non-array action
        action = 0.5
        signal = agent._interpret_action(action)
        assert signal.action == "hold"

    def test_load_model_success(self):
        """Test successful model loading from zip file."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.5, 0.1, 0.2, 0.3]), None)
        mock_ppo_class = MagicMock()
        mock_ppo_class.load.return_value = mock_model

        with (
            patch("os.path.exists", return_value=True),
            patch.dict("sys.modules", {"stable_baselines3": MagicMock(PPO=mock_ppo_class)}),
        ):
            agent = FinRLAgent(model_path="test_model.zip", fallback=True)
            assert agent.model is not None
            assert agent.is_available
            mock_ppo_class.load.assert_called_once_with("test_model.zip")

    def test_predict_with_loaded_model(self):
        """Test predict method with a loaded model."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock()
        # Return a buy signal
        mock_model.predict.return_value = (np.array([0.8, 0.1, 0.2, 0.3]), None)
        mock_ppo_class = MagicMock()
        mock_ppo_class.load.return_value = mock_model

        with (
            patch("os.path.exists", return_value=True),
            patch.dict("sys.modules", {"stable_baselines3": MagicMock(PPO=mock_ppo_class)}),
        ):
            agent = FinRLAgent(model_path="test_model.zip", fallback=True)
            state = np.array([1.0, 2.0, 3.0])
            signal = agent.predict(state)

            assert signal.action == "buy"
            assert signal.confidence > 0
            assert "rsi_buy_threshold" in signal.suggested_params
            mock_model.predict.assert_called_once()

    def test_load_model_exception_with_fallback(self):
        """Test model loading exception with fallback enabled."""
        from unittest.mock import MagicMock, patch

        mock_ppo_class = MagicMock()
        mock_ppo_class.load.side_effect = Exception("Load failed")

        with (
            patch("os.path.exists", return_value=True),
            patch.dict("sys.modules", {"stable_baselines3": MagicMock(PPO=mock_ppo_class)}),
        ):
            agent = FinRLAgent(model_path="test_model.zip", fallback=True)
            assert agent.model is None
            assert not agent.is_available

    def test_predict_exception_handling(self):
        """Test predict handles exceptions gracefully."""
        from unittest.mock import MagicMock, patch

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        mock_ppo_class = MagicMock()
        mock_ppo_class.load.return_value = mock_model

        with (
            patch("os.path.exists", return_value=True),
            patch.dict("sys.modules", {"stable_baselines3": MagicMock(PPO=mock_ppo_class)}),
        ):
            agent = FinRLAgent(model_path="test_model.zip", fallback=True)
            state = np.array([1.0, 2.0, 3.0])
            signal = agent.predict(state)

            assert signal.action == "hold"
            assert signal.confidence == 0.0
