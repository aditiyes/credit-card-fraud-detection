from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect() -> sqlite3.Connection:
    database_path = Path(os.getenv("DATABASE_PATH", "data/predictions.db"))
    if str(database_path) != ":memory:":
        database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            amount REAL NOT NULL,
            fraud_probability REAL NOT NULL,
            decision TEXT NOT NULL,
            mode TEXT NOT NULL,
            explanations TEXT NOT NULL
        )
        """
    )
    return connection


def save_prediction(
    amount: float,
    probability: float,
    decision: str,
    mode: str,
    explanations: list[dict[str, Any]],
) -> int:
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions
                (created_at, amount, fraud_probability, decision, mode, explanations)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                amount,
                probability,
                decision,
                mode,
                json.dumps(explanations),
            ),
        )
        return int(cursor.lastrowid)


def recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, amount, fraud_probability, decision, mode, explanations
            FROM predictions ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "explanations": json.loads(row["explanations"]),
        }
        for row in rows
    ]
