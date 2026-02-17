import requests
from loguru import logger

FEAR_GREED_API = "https://api.alternative.me/fng/?limit=1"


def get_fear_greed_index() -> int:
    """Fetch Bitcoin Fear & Greed Index. Returns 0-100 (0=extreme fear, 100=extreme greed)."""
    try:
        response = requests.get(FEAR_GREED_API, timeout=10)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        logger.debug(f"Fear & Greed Index: {value}")
        return value
    except Exception as e:
        logger.warning(f"Failed to fetch Fear & Greed Index: {e}, using neutral fallback")
        return 50
