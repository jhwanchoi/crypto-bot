import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import _substitute_env, load_config


def test_substitute_env_string(monkeypatch):
    """Test environment variable substitution in strings."""
    monkeypatch.setenv("TEST_VAR", "test_value")
    result = _substitute_env("prefix_${TEST_VAR}_suffix")
    assert result == "prefix_test_value_suffix"


def test_substitute_env_missing_var():
    """Test that missing environment variables are not substituted."""
    result = _substitute_env("prefix_${MISSING_VAR}_suffix")
    assert result == "prefix_${MISSING_VAR}_suffix"


def test_substitute_env_dict(monkeypatch):
    """Test environment variable substitution in dictionaries."""
    monkeypatch.setenv("API_KEY", "secret123")
    monkeypatch.setenv("BASE_URL", "https://api.example.com")

    config = {"api": {"key": "${API_KEY}", "url": "${BASE_URL}"}, "timeout": 30}

    result = _substitute_env(config)
    assert result["api"]["key"] == "secret123"
    assert result["api"]["url"] == "https://api.example.com"
    assert result["timeout"] == 30


def test_substitute_env_list(monkeypatch):
    """Test environment variable substitution in lists."""
    monkeypatch.setenv("HOST1", "server1.com")
    monkeypatch.setenv("HOST2", "server2.com")

    config = ["${HOST1}", "${HOST2}", "static.com"]
    result = _substitute_env(config)

    assert result == ["server1.com", "server2.com", "static.com"]


def test_substitute_env_nested(monkeypatch):
    """Test environment variable substitution in nested structures."""
    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_PORT", "5432")

    config = {
        "database": {
            "host": "${DB_HOST}",
            "port": "${DB_PORT}",
            "options": ["${DB_HOST}:${DB_PORT}"],
        }
    }

    result = _substitute_env(config)
    assert result["database"]["host"] == "localhost"
    assert result["database"]["port"] == "5432"
    assert result["database"]["options"][0] == "localhost:5432"


def test_substitute_env_non_string():
    """Test that non-string values are returned unchanged."""
    assert _substitute_env(123) == 123
    assert _substitute_env(45.67) == 45.67
    assert _substitute_env(True) is True
    assert _substitute_env(None) is None


def test_load_config_basic(tmp_path):
    """Test loading a basic YAML config."""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
api:
  timeout: 30
  retries: 3
trading:
  max_amount: 50000
  tickers:
    - KRW-BTC
    - KRW-ETH
""")

    result = load_config(str(config_file))

    assert result["api"]["timeout"] == 30
    assert result["api"]["retries"] == 3
    assert result["trading"]["max_amount"] == 50000
    assert result["trading"]["tickers"] == ["KRW-BTC", "KRW-ETH"]


def test_load_config_with_env_substitution(tmp_path, monkeypatch):
    """Test loading config with environment variable substitution."""
    monkeypatch.setenv("UPBIT_ACCESS_KEY", "my_access_key")
    monkeypatch.setenv("UPBIT_SECRET_KEY", "my_secret_key")
    monkeypatch.setenv("API_TIMEOUT", "60")

    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
upbit:
  access_key: ${UPBIT_ACCESS_KEY}
  secret_key: ${UPBIT_SECRET_KEY}
api:
  timeout: ${API_TIMEOUT}
  base_url: https://api.upbit.com
""")

    result = load_config(str(config_file))

    assert result["upbit"]["access_key"] == "my_access_key"
    assert result["upbit"]["secret_key"] == "my_secret_key"
    assert result["api"]["timeout"] == "60"
    assert result["api"]["base_url"] == "https://api.upbit.com"


def test_load_config_with_missing_env_vars(tmp_path):
    """Test loading config with missing environment variables."""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
api:
  key: ${MISSING_KEY}
  url: https://api.example.com
""")

    result = load_config(str(config_file))

    # Missing env vars should remain as-is
    assert result["api"]["key"] == "${MISSING_KEY}"
    assert result["api"]["url"] == "https://api.example.com"


def test_load_config_empty_file(tmp_path):
    """Test loading an empty YAML config."""
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("")

    result = load_config(str(config_file))
    assert result is None


def test_load_config_complex_structure(tmp_path, monkeypatch):
    """Test loading a complex nested config with env vars."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
environment: ${ENV}
logging:
  level: ${LOG_LEVEL}
  handlers:
    - type: file
      path: logs/${ENV}.log
    - type: console
database:
  connections:
    - host: db1.${ENV}.com
      port: 5432
    - host: db2.${ENV}.com
      port: 5432
""")

    result = load_config(str(config_file))

    assert result["environment"] == "production"
    assert result["logging"]["level"] == "INFO"
    assert result["logging"]["handlers"][0]["path"] == "logs/production.log"
    assert result["database"]["connections"][0]["host"] == "db1.production.com"
    assert result["database"]["connections"][1]["host"] == "db2.production.com"
