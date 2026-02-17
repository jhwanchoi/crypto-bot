from dataclasses import dataclass, field


@dataclass
class RLSignal:
    """Normalized signal from RL agent."""

    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    suggested_params: dict = field(default_factory=dict)


def normalize_signal(raw: dict) -> RLSignal:
    """Normalize raw RL output into RLSignal."""
    action = raw.get("action", "hold")
    if action not in ("buy", "sell", "hold"):
        action = "hold"
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))
    params = raw.get("suggested_params", {})
    return RLSignal(action=action, confidence=confidence, suggested_params=params)
