# Crypto Trading Bot

3-layer hybrid auto-trading bot for Upbit (KRW market) targeting BTC and ETH.

## Architecture

```
Layer 3: Claude Opus 4.6 (Bedrock)  ─── meta-advisor, market analysis
Layer 2: FinRL (Pretrained PPO)     ─── quantitative signal optimization
Layer 1: DCA + RSI Engine           ─── rule-based execution (always-on fallback)
```

Each layer can override the one below it. If an upper layer fails, the bot falls back to the next layer down.

## Features

- **DCA + RSI Strategy**: Tiered buy multipliers (oversold 1.5x, low 1.2x, neutral 1.0x), RSI-based sell signals
- **Risk Management**: Per-trade limits (5%), daily trade caps (6), stop-loss (8%), take-profit (15%), min cash reserve (20%)
- **Paper Trading**: Simulated orders with real market data and 0.05% fee modeling
- **LLM Advisor**: Claude Opus 4.6 via AWS Bedrock for market phase analysis and parameter tuning
- **RL Agent**: FinRL pretrained weights for quantitative signal generation
- **Notifications**: Telegram alerts for trades and errors
- **Backtesting**: Historical simulation with Sharpe ratio, max drawdown, win rate metrics
- **Trade Logging**: SQLite database for all trade history

## Quick Start

```bash
# Install dependencies
uv sync

# Configure
cp config/settings.yaml config/settings.local.yaml
# Edit config/settings.yaml with your API keys (or set env vars)

# Run in paper mode
uv run python -m src.main
```

## Configuration

Set environment variables or edit `config/settings.yaml`:

```bash
export UPBIT_ACCESS_KEY=your_key
export UPBIT_SECRET_KEY=your_secret
export TELEGRAM_BOT_TOKEN=your_token    # optional
export TELEGRAM_CHAT_ID=your_chat_id    # optional
```

Key settings in `config/settings.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `mode` | `paper` | `paper` or `live` |
| `strategy.interval_hours` | `4` | Trading cycle interval |
| `strategy.base_amount_krw` | `5000` | Base buy amount per trade |
| `risk.stop_loss_pct` | `8` | Stop-loss threshold (%) |
| `risk.take_profit_pct` | `15` | Take-profit threshold (%) |
| `llm.enabled` | `true` | Enable Claude advisor |
| `rl.enabled` | `true` | Enable RL agent |

## Project Structure

```
src/
├── main.py                 # TradingBot orchestrator + APScheduler
├── strategy/
│   ├── dca_rsi.py          # DCA + RSI strategy with apply_overrides
│   └── indicators.py       # RSI (Wilder), Bollinger Bands, price changes
├── risk/
│   └── manager.py          # Buy checks, stop-loss, take-profit
├── exchange/
│   └── upbit_client.py     # Upbit wrapper with paper trading
├── rl/
│   ├── finrl_agent.py      # FinRL pretrained model wrapper
│   └── signal.py           # RL signal dataclass + normalization
├── llm/
│   ├── bedrock_client.py   # AWS Bedrock boto3 client
│   ├── advisor.py          # LLM advisor with JSON parsing + clamping
│   └── prompts.py          # Analysis prompt template
├── data/
│   ├── collector.py        # Market data aggregation
│   └── sentiment.py        # Fear & Greed Index
├── notification/
│   └── telegram.py         # Telegram notifier
└── utils/
    ├── config.py           # YAML loader with env var substitution
    ├── db.py               # SQLite trade database
    └── logger.py           # Loguru setup

backtest/
└── backtester.py           # DCA+RSI historical backtester

tests/                      # 167 tests, 98% coverage
config/
└── settings.yaml           # Bot configuration
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov=backtest --cov-report=term-missing
```

## Toolchain

- **uv** — package management
- **ruff** — linting + formatting
- **pre-commit** — ruff hooks on commit
- **pytest** — testing with pytest-cov

## How It Works

Every 4 hours, the bot runs a cycle per asset (BTC, ETH):

1. **Collect** market data (OHLCV, RSI, Bollinger Bands, Fear & Greed)
2. **RL signal** — FinRL model suggests parameter adjustments
3. **LLM analysis** — Claude analyzes market phase, may override action
4. **Strategy** — DCA+RSI determines buy/sell/skip with adjusted parameters
5. **Risk check** — validates against daily limits, cash reserve, position size
6. **Execute** — places order (paper or live)
7. **Log & notify** — records trade to SQLite, sends Telegram alert

## License

MIT
