import os
import re

import yaml


def _substitute_env(value):
    """Recursively substitute ${ENV_VAR} patterns in config values."""
    if isinstance(value, str):
        pattern = r"\$\{(\w+)\}"

        def replacer(match):
            return os.environ.get(match.group(1), match.group(0))

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load YAML config with environment variable substitution."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return _substitute_env(config)
