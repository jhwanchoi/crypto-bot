import pandas as pd


def calculate_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method."""
    delta = closes.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Handle division by zero: when avg_loss is 0, RSI should be 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 (all gains), set RSI to 100
    rsi = rsi.where(avg_loss != 0, 100.0)
    # When avg_gain is 0 (all losses), set RSI to 0
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi


def calculate_bollinger_bands(
    closes: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands. Returns (upper, middle, lower)."""
    middle = closes.rolling(window=period).mean()
    std = closes.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def get_bb_position(close: float, upper: float, lower: float) -> str:
    """Determine price position relative to Bollinger Bands."""
    if pd.isna(upper) or pd.isna(lower):
        return "unknown"
    if close >= upper:
        return "above_upper"
    elif close <= lower:
        return "below_lower"
    else:
        mid = (upper + lower) / 2
        if close > mid:
            return "upper_half"
        return "lower_half"


def calculate_price_change(closes: pd.Series, periods: int) -> float:
    """Calculate percentage price change over N periods."""
    if len(closes) < periods + 1:
        return 0.0
    return (closes.iloc[-1] - closes.iloc[-1 - periods]) / closes.iloc[-1 - periods]
