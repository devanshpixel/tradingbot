from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str  # BUY (long) or SHORT
    qty: int
    entry: float
    stop: float
    target: float
    confidence: float
    status: str  # OPEN/CLOSED
    opened_at: str
    closed_at: str | None
    exit_price: float | None
    pnl: float | None
    meta_json: str


class TradeTracker:
    def __init__(self, db_path: str = "paper_trades.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty INTEGER NOT NULL,
                    entry REAL NOT NULL,
                    stop REAL NOT NULL,
                    target REAL NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    exit_price REAL,
                    pnl REAL,
                    meta_json TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);")

    def open_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry: float,
        stop: float,
        target: float,
        confidence: float,
        meta: dict[str, Any] | None = None,
    ) -> None:
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades(symbol, side, qty, entry, stop, target, confidence, status, opened_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?);
                """,
                (symbol, side, int(qty), float(entry), float(stop), float(target), float(confidence), _utc_now(), meta_json),
            )

    def close_trade(self, trade_id: int, exit_price: float) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trades WHERE id = ? AND status = 'OPEN';", (int(trade_id),)).fetchone()
            if not row:
                return
            entry = float(row["entry"])
            qty = int(row["qty"])
            side = str(row["side"])
            if side == "BUY":
                pnl = (float(exit_price) - entry) * qty
            else:
                pnl = (entry - float(exit_price)) * qty

            conn.execute(
                """
                UPDATE trades
                SET status = 'CLOSED', closed_at = ?, exit_price = ?, pnl = ?
                WHERE id = ?;
                """,
                (_utc_now(), float(exit_price), float(pnl), int(trade_id)),
            )

    def list_open_trades(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, symbol, side, qty, entry, stop, target, confidence, opened_at FROM trades WHERE status='OPEN' ORDER BY opened_at DESC;"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
            columns=["id", "symbol", "side", "qty", "entry", "stop", "target", "confidence", "opened_at"]
        )

    def trade_history(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, symbol, side, qty, entry, stop, target, confidence, status, opened_at, closed_at, exit_price, pnl FROM trades ORDER BY opened_at DESC;"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(
            columns=[
                "id",
                "symbol",
                "side",
                "qty",
                "entry",
                "stop",
                "target",
                "confidence",
                "status",
                "opened_at",
                "closed_at",
                "exit_price",
                "pnl",
            ]
        )

    def stats(self) -> dict[str, Any]:
        df = self.trade_history()
        closed = df[df["status"] == "CLOSED"] if not df.empty else df
        if closed.empty:
            return {"trades_closed": 0, "win_rate": None, "total_pnl": 0.0, "avg_pnl": None}
        wins = (closed["pnl"] > 0).sum()
        total = int(len(closed))
        return {
            "trades_closed": total,
            "win_rate": round(float(wins / total * 100.0), 2) if total else None,
            "total_pnl": round(float(closed["pnl"].sum()), 2),
            "avg_pnl": round(float(closed["pnl"].mean()), 2),
        }

