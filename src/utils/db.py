import sqlite3
from datetime import datetime
from pathlib import Path


class TradeDB:
    """SQLite wrapper for trade history."""

    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def init(self):
        """Create trades table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
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

    def insert_trade(
        self,
        ticker: str,
        side: str,
        price: float,
        amount: float,
        total_krw: float,
        rsi: float | None = None,
        rl_signal: str | None = None,
        llm_reasoning: str | None = None,
    ):
        """Insert a new trade record."""
        timestamp = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (timestamp, ticker, side, price, amount, total_krw, rsi, rl_signal, llm_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (timestamp, ticker, side, price, amount, total_krw, rsi, rl_signal, llm_reasoning),
            )
            conn.commit()

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        """Get recent trades, ordered by id DESC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_daily_trade_count(self) -> int:
        """Count trades made today."""
        today = datetime.now().date().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE DATE(timestamp) = ?", (today,))
            return cursor.fetchone()[0]

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary with total invested and trades per ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT
                    ticker,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN side = 'buy' THEN total_krw ELSE 0 END) as total_invested
                FROM trades
                GROUP BY ticker
            """
            )
            rows = cursor.fetchall()

            summary = {"total_invested": 0, "total_trades": {}}

            for row in rows:
                ticker, trade_count, invested = row
                summary["total_trades"][ticker] = trade_count
                summary["total_invested"] += invested or 0

            return summary
