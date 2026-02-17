import os
import sys

from loguru import logger


def setup_logger(
    level: str = "INFO", log_file: str = "logs/crypto_bot.log", rotation: str = "10 MB"
) -> logger:
    """Configure and return loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}"
    )
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(log_file, level=level, rotation=rotation, retention="30 days")
    return logger
