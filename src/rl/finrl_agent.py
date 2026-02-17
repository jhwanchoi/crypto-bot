import os

import numpy as np
from loguru import logger

from src.rl.signal import RLSignal


class FinRLAgent:
    """Wrapper for FinRL pretrained model with fallback."""

    def __init__(self, model_path: str = "models/finrl_pretrained.zip", fallback: bool = True):
        self.model_path = model_path
        self.fallback = fallback
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            if self.fallback:
                logger.warning(f"RL model not found at {self.model_path}, using fallback mode")
                return
            raise FileNotFoundError(f"RL model not found: {self.model_path}")
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(self.model_path)
            logger.info(f"RL model loaded from {self.model_path}")
        except Exception as e:
            if self.fallback:
                logger.warning(f"Failed to load RL model: {e}, using fallback mode")
            else:
                raise

    def predict(self, state: np.ndarray) -> RLSignal:
        """Generate trading signal from market state."""
        if self.model is None:
            return RLSignal(action="hold", confidence=0.0, suggested_params={})

        try:
            action, _states = self.model.predict(state, deterministic=True)
            return self._interpret_action(action)
        except Exception as e:
            logger.error(f"RL prediction failed: {e}")
            return RLSignal(action="hold", confidence=0.0, suggested_params={})

    def _interpret_action(self, action) -> RLSignal:
        """Interpret raw model action into RLSignal."""
        if isinstance(action, np.ndarray):
            action = action.flatten()
            if len(action) >= 4:
                # Continuous action space: [trade_direction, rsi_threshold_adj, buy_mult_adj, tp_adj]
                direction = float(action[0])
                rsi_adj = float(action[1])
                mult_adj = float(action[2])
                tp_adj = float(action[3])

                if direction > 0.3:
                    act = "buy"
                elif direction < -0.3:
                    act = "sell"
                else:
                    act = "hold"

                confidence = min(1.0, abs(direction))
                return RLSignal(
                    action=act,
                    confidence=confidence,
                    suggested_params={
                        "rsi_buy_threshold": max(20, min(40, 30 + rsi_adj * 10)),
                        "buy_multiplier": max(0.5, min(2.5, 1.0 + mult_adj)),
                        "take_profit_pct": max(5, min(30, 15 + tp_adj * 10)),
                    },
                )

        return RLSignal(action="hold", confidence=0.0, suggested_params={})

    @property
    def is_available(self) -> bool:
        return self.model is not None
