# Crypto Trading Bot - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upbit KRW 마켓에서 BTC+ETH를 자동 매매하는 3계층 하이브리드 봇 구현

**Architecture:** DCA+RSI 규칙 기반 실행 엔진(Layer 1) + FinRL pretrained 정량 시그널(Layer 2) + Claude Opus 4.6 Bedrock 메타 어드바이저(Layer 3). 각 레이어는 독립적이며 상위 레이어 장애 시 하위 레이어가 기본값으로 독립 운영.

**Tech Stack:** Python, pyupbit, FinRL, stable-baselines3, boto3(Bedrock), pandas, APScheduler, SQLite, python-telegram-bot, loguru, PyYAML

**Design Doc:** `docs/plans/2026-02-17-crypto-bot-design.md`

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/settings.yaml`
- Create: all `__init__.py` files
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/exchange/__init__.py`, `src/strategy/__init__.py`, `src/rl/__init__.py`, `src/llm/__init__.py`, `src/risk/__init__.py`, `src/data/__init__.py`, `src/notification/__init__.py`, `src/utils/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p config src/{exchange,strategy,rl,llm,risk,data,notification,utils} data models backtest tests
```

**Step 2: Create requirements.txt**

```
pyupbit>=0.2.33
pandas>=2.0.0
numpy>=1.24.0
PyYAML>=6.0
APScheduler>=3.10.0
loguru>=0.7.0
python-telegram-bot>=20.0
boto3>=1.28.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
finrl>=0.3.6
requests>=2.31.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

**Step 3: Create config/settings.yaml**

```yaml
exchange:
  name: upbit
  access_key: ${UPBIT_ACCESS_KEY}
  secret_key: ${UPBIT_SECRET_KEY}

assets:
  - ticker: KRW-BTC
    allocation: 0.6
  - ticker: KRW-ETH
    allocation: 0.4

strategy:
  type: dca_rsi
  interval_hours: 4
  base_amount_krw: 5000
  rsi_period: 14
  rsi_buy_threshold: 30
  rsi_sell_threshold: 70
  buy_multipliers:
    oversold: 1.5      # RSI < 30
    low: 1.2            # RSI 30-45
    neutral: 1.0        # RSI 45-55

risk:
  max_buy_per_trade_pct: 5
  max_daily_trades: 6
  stop_loss_pct: 8
  take_profit_pct: 15
  take_profit_sell_ratio: 0.3
  max_position_pct: 80
  min_cash_pct: 20

rl:
  enabled: true
  model_path: models/finrl_pretrained.zip
  fallback_to_default: true

llm:
  enabled: true
  provider: bedrock
  model_id: anthropic.claude-opus-4-6-v1
  temperature: 0
  max_tokens: 1024
  region: us-east-1
  fallback_to_default: true

notification:
  telegram:
    enabled: false
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}

logging:
  level: INFO
  file: logs/crypto_bot.log
  rotation: 10 MB

mode: paper  # paper | live | backtest
```

**Step 4: Create all __init__.py files and .gitignore**

**Step 5: Install dependencies and verify**

```bash
pip install -r requirements.txt
python -c "import pyupbit, pandas, loguru, boto3; print('OK')"
```

**Step 6: Initialize git and commit**

```bash
git init
git add .
git commit -m "chore: project scaffolding with dependencies and config"
```

---

### Task 2: Logger & Database Utilities

**Files:**
- Create: `src/utils/logger.py`
- Create: `src/utils/db.py`
- Test: `tests/test_utils.py`

**Step 1: Write failing tests**

```python
# tests/test_utils.py
import os
import sqlite3
from src.utils.logger import setup_logger
from src.utils.db import TradeDB

def test_setup_logger_returns_logger():
    logger = setup_logger()
    assert logger is not None

def test_trade_db_create_tables(tmp_path):
    db_path = tmp_path / "test.db"
    db = TradeDB(str(db_path))
    db.init()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "trades" in tables

def test_trade_db_insert_and_query(tmp_path):
    db_path = tmp_path / "test.db"
    db = TradeDB(str(db_path))
    db.init()
    db.insert_trade(
        ticker="KRW-BTC", side="buy", price=50000000.0,
        amount=0.001, total_krw=50000.0, rsi=28.5,
        rl_signal="buy", llm_reasoning="oversold"
    )
    trades = db.get_recent_trades(limit=1)
    assert len(trades) == 1
    assert trades[0]["ticker"] == "KRW-BTC"

def test_trade_db_daily_trade_count(tmp_path):
    db_path = tmp_path / "test.db"
    db = TradeDB(str(db_path))
    db.init()
    db.insert_trade("KRW-BTC", "buy", 50000000, 0.001, 50000, 28.5, "buy", "test")
    db.insert_trade("KRW-ETH", "buy", 3000000, 0.01, 30000, 32.0, "buy", "test")
    assert db.get_daily_trade_count() == 2
```

**Step 2: Implement logger.py**

```python
# src/utils/logger.py
from loguru import logger
import sys

def setup_logger(level="INFO", log_file="logs/crypto_bot.log", rotation="10 MB"):
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
    logger.add(log_file, level=level, rotation=rotation, retention="30 days")
    return logger
```

**Step 3: Implement db.py**

```python
# src/utils/db.py
import sqlite3
from datetime import datetime, date

class TradeDB:
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path

    def init(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                total_krw REAL NOT NULL,
                rsi REAL,
                rl_signal TEXT,
                llm_reasoning TEXT
            )
        """)
        conn.commit()
        conn.close()

    def insert_trade(self, ticker, side, price, amount, total_krw, rsi=None, rl_signal=None, llm_reasoning=None):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO trades (timestamp, ticker, side, price, amount, total_krw, rsi, rl_signal, llm_reasoning) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), ticker, side, price, amount, total_krw, rsi, rl_signal, llm_reasoning)
        )
        conn.commit()
        conn.close()

    def get_recent_trades(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    def get_daily_trade_count(self):
        conn = sqlite3.connect(self.db_path)
        today = date.today().isoformat()
        cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?", (f"{today}%",))
        count = cursor.fetchone()[0]
        conn.close()
        return count
```

**Step 4: Run tests**

```bash
pytest tests/test_utils.py -v
```

**Step 5: Commit**

```bash
git add src/utils/ tests/test_utils.py
git commit -m "feat: add logger and trade database utilities"
```

---

### Task 3: Config Loader

**Files:**
- Create: `src/utils/config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_config.py
from src.utils.config import load_config

def test_load_config(tmp_path):
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("""
exchange:
  name: upbit
assets:
  - ticker: KRW-BTC
    allocation: 0.6
strategy:
  type: dca_rsi
  interval_hours: 4
mode: paper
""")
    config = load_config(str(cfg_file))
    assert config["exchange"]["name"] == "upbit"
    assert config["assets"][0]["ticker"] == "KRW-BTC"
    assert config["mode"] == "paper"

def test_load_config_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_ACCESS_KEY", "test_key")
    cfg_file = tmp_path / "settings.yaml"
    cfg_file.write_text("""
exchange:
  access_key: ${UPBIT_ACCESS_KEY}
""")
    config = load_config(str(cfg_file))
    assert config["exchange"]["access_key"] == "test_key"
```

**Step 2: Implement config.py**

```python
# src/utils/config.py
import os
import re
import yaml

def _substitute_env(value):
    if isinstance(value, str):
        pattern = r'\$\{(\w+)\}'
        def replacer(match):
            return os.environ.get(match.group(1), match.group(0))
        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value

def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return _substitute_env(config)
```

**Step 3: Run tests, commit**

---

### Task 4: Upbit Exchange Client

**Files:**
- Create: `src/exchange/upbit_client.py`
- Test: `tests/test_exchange.py`

**Step 1: Write failing tests (mocked)**

```python
# tests/test_exchange.py
from unittest.mock import patch, MagicMock
from src.exchange.upbit_client import UpbitClient

def test_get_balance():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        mock_upbit = MagicMock()
        mock_upbit.get_balance.return_value = 100000.0
        mock_pyupbit.Upbit.return_value = mock_upbit
        client = UpbitClient("key", "secret")
        assert client.get_balance("KRW") == 100000.0

def test_get_ohlcv():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        import pandas as pd
        mock_df = pd.DataFrame({"open": [100], "high": [110], "low": [90], "close": [105], "volume": [1000]})
        mock_pyupbit.get_ohlcv.return_value = mock_df
        client = UpbitClient("key", "secret")
        df = client.get_ohlcv("KRW-BTC", interval="minute240", count=14)
        assert len(df) == 1

def test_buy_market_order():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        mock_upbit = MagicMock()
        mock_upbit.buy_market_order.return_value = {"uuid": "test-uuid"}
        mock_pyupbit.Upbit.return_value = mock_upbit
        client = UpbitClient("key", "secret")
        result = client.buy_market_order("KRW-BTC", 10000)
        assert result["uuid"] == "test-uuid"

def test_sell_market_order():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        mock_upbit = MagicMock()
        mock_upbit.sell_market_order.return_value = {"uuid": "sell-uuid"}
        mock_pyupbit.Upbit.return_value = mock_upbit
        client = UpbitClient("key", "secret")
        result = client.sell_market_order("KRW-BTC", 0.001)
        assert result["uuid"] == "sell-uuid"

def test_get_current_price():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        mock_pyupbit.get_current_price.return_value = 50000000
        client = UpbitClient("key", "secret")
        assert client.get_current_price("KRW-BTC") == 50000000

def test_get_avg_buy_price():
    with patch("src.exchange.upbit_client.pyupbit") as mock_pyupbit:
        mock_upbit = MagicMock()
        mock_upbit.get_avg_buy_price.return_value = 48000000.0
        mock_pyupbit.Upbit.return_value = mock_upbit
        client = UpbitClient("key", "secret")
        assert client.get_avg_buy_price("BTC") == 48000000.0
```

**Step 2: Implement upbit_client.py**

Wraps pyupbit with error handling and paper trading mode support.

**Step 3: Run tests, commit**

---

### Task 5: Technical Indicators (RSI, Bollinger Bands)

**Files:**
- Create: `src/strategy/indicators.py`
- Test: `tests/test_indicators.py`

**Step 1: Write failing tests**

```python
# tests/test_indicators.py
import pandas as pd
import numpy as np
from src.strategy.indicators import calculate_rsi, calculate_bollinger_bands

def test_rsi_returns_series():
    closes = pd.Series([44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
                        46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41,
                        46.22, 45.64])
    rsi = calculate_rsi(closes, period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(closes)

def test_rsi_range():
    np.random.seed(42)
    closes = pd.Series(np.random.uniform(100, 200, 100))
    rsi = calculate_rsi(closes, period=14)
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()

def test_rsi_oversold_on_declining():
    closes = pd.Series([100 - i * 2 for i in range(20)])
    rsi = calculate_rsi(closes, period=14)
    assert rsi.iloc[-1] < 30

def test_bollinger_bands_structure():
    closes = pd.Series(np.random.uniform(100, 200, 30))
    upper, middle, lower = calculate_bollinger_bands(closes, period=20, std_dev=2)
    assert len(upper) == len(closes)
    valid_idx = ~upper.isna()
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (lower[valid_idx] <= middle[valid_idx]).all()
```

**Step 2: Implement indicators.py**

RSI via Wilder smoothing method. Bollinger Bands via rolling mean +/- std.

**Step 3: Run tests, commit**

---

### Task 6: Risk Manager

**Files:**
- Create: `src/risk/manager.py`
- Test: `tests/test_risk.py`

**Step 1: Write failing tests**

```python
# tests/test_risk.py
from src.risk.manager import RiskManager

def test_check_buy_allowed_within_limits():
    rm = RiskManager(
        max_buy_per_trade_pct=5, max_daily_trades=6,
        stop_loss_pct=8, take_profit_pct=15,
        take_profit_sell_ratio=0.3, max_position_pct=80, min_cash_pct=20
    )
    allowed, reason = rm.check_buy_allowed(
        total_seed=500000, current_cash=200000,
        buy_amount=25000, daily_trade_count=2
    )
    assert allowed is True

def test_check_buy_blocked_daily_limit():
    rm = RiskManager(
        max_buy_per_trade_pct=5, max_daily_trades=6,
        stop_loss_pct=8, take_profit_pct=15,
        take_profit_sell_ratio=0.3, max_position_pct=80, min_cash_pct=20
    )
    allowed, reason = rm.check_buy_allowed(
        total_seed=500000, current_cash=200000,
        buy_amount=25000, daily_trade_count=6
    )
    assert allowed is False
    assert "daily" in reason.lower()

def test_check_buy_blocked_min_cash():
    rm = RiskManager(
        max_buy_per_trade_pct=5, max_daily_trades=6,
        stop_loss_pct=8, take_profit_pct=15,
        take_profit_sell_ratio=0.3, max_position_pct=80, min_cash_pct=20
    )
    allowed, reason = rm.check_buy_allowed(
        total_seed=500000, current_cash=110000,
        buy_amount=25000, daily_trade_count=0
    )
    assert allowed is False
    assert "cash" in reason.lower()

def test_check_stop_loss():
    rm = RiskManager(
        max_buy_per_trade_pct=5, max_daily_trades=6,
        stop_loss_pct=8, take_profit_pct=15,
        take_profit_sell_ratio=0.3, max_position_pct=80, min_cash_pct=20
    )
    assert rm.should_stop_loss(avg_price=50000000, current_price=45000000) is True
    assert rm.should_stop_loss(avg_price=50000000, current_price=48000000) is False

def test_check_take_profit():
    rm = RiskManager(
        max_buy_per_trade_pct=5, max_daily_trades=6,
        stop_loss_pct=8, take_profit_pct=15,
        take_profit_sell_ratio=0.3, max_position_pct=80, min_cash_pct=20
    )
    should, sell_ratio = rm.should_take_profit(avg_price=50000000, current_price=58000000)
    assert should is True
    assert sell_ratio == 0.3
```

**Step 2: Implement manager.py**

**Step 3: Run tests, commit**

---

### Task 7: DCA + RSI Strategy Engine

**Files:**
- Create: `src/strategy/dca_rsi.py`
- Test: `tests/test_strategy.py`

**Step 1: Write failing tests**

```python
# tests/test_strategy.py
from src.strategy.dca_rsi import DcaRsiStrategy

def test_determine_action_oversold():
    strategy = DcaRsiStrategy(
        base_amount=5000, rsi_buy_threshold=30, rsi_sell_threshold=70,
        buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )
    action = strategy.determine_action(rsi=25.0, has_position=False)
    assert action["type"] == "buy"
    assert action["amount"] == 7500  # 5000 * 1.5

def test_determine_action_neutral():
    strategy = DcaRsiStrategy(
        base_amount=5000, rsi_buy_threshold=30, rsi_sell_threshold=70,
        buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )
    action = strategy.determine_action(rsi=50.0, has_position=False)
    assert action["type"] == "buy"
    assert action["amount"] == 5000

def test_determine_action_skip():
    strategy = DcaRsiStrategy(
        base_amount=5000, rsi_buy_threshold=30, rsi_sell_threshold=70,
        buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )
    action = strategy.determine_action(rsi=62.0, has_position=False)
    assert action["type"] == "skip"

def test_determine_action_overbought_sell():
    strategy = DcaRsiStrategy(
        base_amount=5000, rsi_buy_threshold=30, rsi_sell_threshold=70,
        buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )
    action = strategy.determine_action(rsi=75.0, has_position=True)
    assert action["type"] == "sell"

def test_apply_llm_overrides():
    strategy = DcaRsiStrategy(
        base_amount=5000, rsi_buy_threshold=30, rsi_sell_threshold=70,
        buy_multipliers={"oversold": 1.5, "low": 1.2, "neutral": 1.0}
    )
    overrides = {"rsi_buy_threshold": 25, "rsi_sell_threshold": 75, "buy_multiplier": 1.8}
    strategy.apply_overrides(overrides)
    assert strategy.rsi_buy_threshold == 25
    assert strategy.rsi_sell_threshold == 75
```

**Step 2: Implement dca_rsi.py**

**Step 3: Run tests, commit**

---

### Task 8: FinRL Agent Wrapper

**Files:**
- Create: `src/rl/finrl_agent.py`
- Create: `src/rl/signal.py`
- Test: `tests/test_rl.py`

**Step 1: Write failing tests**

```python
# tests/test_rl.py
import numpy as np
from src.rl.signal import RLSignal, normalize_signal
from src.rl.finrl_agent import FinRLAgent

def test_rl_signal_dataclass():
    sig = RLSignal(action="buy", confidence=0.72, suggested_params={
        "rsi_buy_threshold": 28, "buy_multiplier": 1.3
    })
    assert sig.action == "buy"
    assert sig.confidence == 0.72

def test_normalize_signal_clamps_confidence():
    sig = normalize_signal({"action": "buy", "confidence": 1.5, "suggested_params": {}})
    assert sig.confidence <= 1.0

def test_finrl_agent_fallback_when_no_model():
    agent = FinRLAgent(model_path="nonexistent.zip", fallback=True)
    signal = agent.predict(np.zeros(10))
    assert signal.action == "hold"
    assert signal.confidence == 0.0
```

**Step 2: Implement finrl_agent.py with fallback logic and signal.py dataclass**

**Step 3: Run tests, commit**

---

### Task 9: Bedrock Client & LLM Advisor

**Files:**
- Create: `src/llm/bedrock_client.py`
- Create: `src/llm/advisor.py`
- Create: `src/llm/prompts.py`
- Test: `tests/test_advisor.py`

**Step 1: Write failing tests (mocked boto3)**

```python
# tests/test_advisor.py
import json
from unittest.mock import patch, MagicMock
from src.llm.advisor import LLMAdvisor
from src.llm.prompts import build_analysis_prompt

def test_build_analysis_prompt():
    prompt = build_analysis_prompt(
        ticker="KRW-BTC", rsi=28.5, price=50000000,
        price_change_5=0.02, price_change_20=-0.05,
        bb_position="below_lower", rl_signal={"action": "buy", "confidence": 0.8},
        fear_greed_index=25
    )
    assert "KRW-BTC" in prompt
    assert "28.5" in prompt

def test_advisor_parse_response():
    advisor = LLMAdvisor(model_id="test", region="us-east-1", enabled=False)
    raw = json.dumps({
        "market_phase": "bearish",
        "rsi_buy_threshold": 28,
        "rsi_sell_threshold": 75,
        "buy_multiplier": 1.3,
        "take_profit_pct": 18,
        "action_override": None,
        "reasoning": "Oversold conditions",
        "risk_alert": None
    })
    result = advisor.parse_response(raw)
    assert result["market_phase"] == "bearish"
    assert result["buy_multiplier"] == 1.3

def test_advisor_fallback_when_disabled():
    advisor = LLMAdvisor(model_id="test", region="us-east-1", enabled=False)
    result = advisor.analyze(ticker="KRW-BTC", rsi=50, price=50000000,
                             price_changes={}, bb_position="middle",
                             rl_signal={}, fear_greed=50)
    assert result is None
```

**Step 2: Implement bedrock_client.py, advisor.py, prompts.py**

Key: bedrock_client wraps boto3 invoke_model, advisor orchestrates prompt building + response parsing, prompts.py contains the analysis prompt template with JSON schema enforcement.

**Step 3: Run tests, commit**

---

### Task 10: Data Collector (Sentiment)

**Files:**
- Create: `src/data/collector.py`
- Create: `src/data/sentiment.py`
- Test: `tests/test_data.py`

**Step 1: Write failing tests**

```python
# tests/test_data.py
from unittest.mock import patch
from src.data.sentiment import get_fear_greed_index
from src.data.collector import MarketDataCollector

def test_fear_greed_returns_int():
    with patch("src.data.sentiment.requests") as mock_req:
        mock_req.get.return_value.json.return_value = {"data": [{"value": "25"}]}
        mock_req.get.return_value.status_code = 200
        result = get_fear_greed_index()
        assert isinstance(result, int)
        assert 0 <= result <= 100

def test_fear_greed_fallback_on_error():
    with patch("src.data.sentiment.requests") as mock_req:
        mock_req.get.side_effect = Exception("API down")
        result = get_fear_greed_index()
        assert result == 50  # neutral fallback

def test_market_data_collector():
    collector = MarketDataCollector()
    # Verify it has the expected interface
    assert hasattr(collector, "collect")
```

**Step 2: Implement sentiment.py (Fear & Greed API) and collector.py (aggregates OHLCV + indicators + sentiment)**

**Step 3: Run tests, commit**

---

### Task 11: Telegram Notification

**Files:**
- Create: `src/notification/telegram.py`
- Test: `tests/test_notification.py`

**Step 1: Write failing tests**

```python
# tests/test_notification.py
from unittest.mock import patch, AsyncMock
from src.notification.telegram import TelegramNotifier

def test_format_trade_message():
    notifier = TelegramNotifier(bot_token="test", chat_id="123", enabled=False)
    msg = notifier.format_trade_message(
        ticker="KRW-BTC", side="buy", price=50000000,
        amount=0.001, total_krw=50000, rsi=28.5, reasoning="Oversold"
    )
    assert "KRW-BTC" in msg
    assert "매수" in msg

def test_format_error_message():
    notifier = TelegramNotifier(bot_token="test", chat_id="123", enabled=False)
    msg = notifier.format_error_message("API connection failed")
    assert "error" in msg.lower() or "오류" in msg
```

**Step 2: Implement telegram.py**

**Step 3: Run tests, commit**

---

### Task 12: Main Orchestrator & Scheduler

**Files:**
- Create: `src/main.py`
- Test: `tests/test_main.py`

**Step 1: Write failing tests**

```python
# tests/test_main.py
from unittest.mock import patch, MagicMock
from src.main import TradingBot

def test_trading_bot_init():
    with patch("src.main.load_config") as mock_cfg:
        mock_cfg.return_value = {
            "exchange": {"access_key": "k", "secret_key": "s"},
            "assets": [{"ticker": "KRW-BTC", "allocation": 0.6}],
            "strategy": {"type": "dca_rsi", "interval_hours": 4, "base_amount_krw": 5000,
                         "rsi_period": 14, "rsi_buy_threshold": 30, "rsi_sell_threshold": 70,
                         "buy_multipliers": {"oversold": 1.5, "low": 1.2, "neutral": 1.0}},
            "risk": {"max_buy_per_trade_pct": 5, "max_daily_trades": 6, "stop_loss_pct": 8,
                     "take_profit_pct": 15, "take_profit_sell_ratio": 0.3,
                     "max_position_pct": 80, "min_cash_pct": 20},
            "rl": {"enabled": False, "model_path": "", "fallback_to_default": True},
            "llm": {"enabled": False, "model_id": "", "region": "", "temperature": 0,
                    "max_tokens": 1024, "fallback_to_default": True},
            "notification": {"telegram": {"enabled": False, "bot_token": "", "chat_id": ""}},
            "logging": {"level": "INFO", "file": "logs/test.log", "rotation": "10 MB"},
            "mode": "paper"
        }
        bot = TradingBot(config_path="dummy")
        assert bot is not None
        assert bot.mode == "paper"
```

**Step 2: Implement main.py**

The TradingBot class:
1. Loads config
2. Initializes all components (exchange, strategy, risk, rl, llm, notification, db)
3. Runs the execution cycle per asset on schedule
4. Execution cycle: collect data → RL signal → LLM analysis → strategy decision → risk check → execute → log → notify

**Step 3: Run tests, commit**

---

### Task 13: Backtester

**Files:**
- Create: `backtest/backtester.py`
- Test: `tests/test_backtest.py`

**Step 1: Write failing tests**

```python
# tests/test_backtest.py
import pandas as pd
import numpy as np
from backtest.backtester import Backtester

def test_backtester_run_returns_results():
    np.random.seed(42)
    prices = pd.Series(np.random.uniform(45000000, 55000000, 100))
    bt = Backtester(initial_cash=500000, base_amount=5000)
    results = bt.run(prices)
    assert "total_return_pct" in results
    assert "total_trades" in results
    assert "sharpe_ratio" in results

def test_backtester_no_negative_cash():
    prices = pd.Series([50000000] * 100)
    bt = Backtester(initial_cash=500000, base_amount=5000)
    results = bt.run(prices)
    assert results["min_cash"] >= 0
```

**Step 2: Implement backtester.py**

Simulates DCA+RSI strategy on historical price data. Tracks cash, positions, trades, returns.

**Step 3: Run tests, commit**

---

### Task 14: Integration Test & Final Verification

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test (paper mode, all layers)**

```python
# tests/test_integration.py
from unittest.mock import patch, MagicMock
from src.main import TradingBot

def test_full_cycle_paper_mode():
    """End-to-end test: data collection → RL → LLM → strategy → execution in paper mode"""
    # Mock all external dependencies, verify the full pipeline runs without error
    ...
```

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 3: Final commit**

```bash
git add .
git commit -m "feat: complete crypto trading bot with 3-layer hybrid architecture"
```
