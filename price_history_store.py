"""
SQLite-based historical price data store.

Stores collected price data from Yahoo Finance and FRED in a SQLite database
for historical tracking and analysis.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PriceHistoryStore:
    """SQLite store for historical price data."""

    def __init__(self, db_path: Path | str):
        """
        Initialize the price history store.

        Args:
            db_path: Path to the SQLite database file
        """
        if isinstance(db_path, str):
            db_path = Path(db_path)
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the price_history table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    name TEXT NOT NULL,
                    symbol TEXT,
                    date TEXT NOT NULL,
                    time TEXT,
                    price REAL,
                    prev_close REAL,
                    change REAL,
                    change_pct REAL,
                    raw_json TEXT,
                    UNIQUE(source, name, date, time)
                )
                """
            )
            conn.commit()
            logger.debug("Ensured price_history table exists")

    def upsert_from_collection(self, collected_data: Dict[str, Any]) -> None:
        """
        Insert or update data from a collection dictionary.

        Expected structure:
        {
            "yahoo_finance": [
                {
                    "name": "S&P 500",
                    "symbol": "^GSPC",
                    "date": "2025-11-13",
                    "time": "12:14:08",
                    "price": 6850.92,
                    "prev_close": 6846.61,
                    "change": 4.31,
                    "change_pct": 0.0629,
                    ...
                },
                ...
            ],
            "fred": [
                {
                    "name": "Federal Funds Target Rate (Upper)",
                    "symbol": "DFEDTARU",
                    "date": "2025-11-13",
                    "time": None,
                    "price": 5.5,
                    ...
                },
                ...
            ]
        }

        Args:
            collected_data: Dictionary with 'yahoo_finance' and/or 'fred' keys
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            inserted = 0
            skipped = 0

            for source, entries in collected_data.items():
                if not isinstance(entries, list):
                    continue

                for entry in entries:
                    if not isinstance(entry, dict):
                        continue

                    name = entry.get("name", "")
                    symbol = entry.get("symbol")
                    date = entry.get("date", "")
                    time = entry.get("time")
                    price = entry.get("price")
                    prev_close = entry.get("prev_close")
                    change = entry.get("change")
                    change_pct = entry.get("change_pct")
                    raw_json = json.dumps(entry, default=str)

                    if not name or not date:
                        logger.warning("Skipping entry with missing name or date: %s", entry)
                        skipped += 1
                        continue

                    try:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO price_history
                            (source, name, symbol, date, time, price, prev_close, change, change_pct, raw_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                source,
                                name,
                                symbol,
                                date,
                                time,
                                price,
                                prev_close,
                                change,
                                change_pct,
                                raw_json,
                            ),
                        )
                        if cursor.rowcount > 0:
                            inserted += 1
                        else:
                            skipped += 1
                    except sqlite3.Error as e:
                        logger.error("Error inserting entry %s: %s", entry, e)
                        skipped += 1

            conn.commit()
            logger.info("Upserted %d entries, skipped %d duplicates", inserted, skipped)

    def get_history(
        self,
        name: str,
        source: str = "yahoo_finance",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 365,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical data for a given name and source.

        Args:
            name: Name of the indicator (e.g., "S&P 500")
            source: Data source ("yahoo_finance" or "fred")
            start_date: Optional start date (YYYY-MM-DD format)
            end_date: Optional end date (YYYY-MM-DD format)
            limit: Maximum number of records to return (default: 365)

        Returns:
            List of dictionaries containing historical records, sorted by date/time ascending
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM price_history WHERE source = ? AND name = ?"
            params: List[Any] = [source, name]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date ASC, time ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = []
            for row in rows:
                record = dict(row)
                # Parse raw_json if it exists
                if record.get("raw_json"):
                    try:
                        record["raw_json"] = json.loads(record["raw_json"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                result.append(record)

            return result

    def insert_rows(self, rows: list[dict]) -> None:
        """
        Bulk insert rows into the price_history table.

        Args:
            rows: List of dictionaries, each with keys:
                source, name, symbol, date, time, price, prev_close, change, change_pct, raw_json
        """
        if not rows:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                inserted = 0

                # Prepare rows for insertion
                prepared_rows = []
                for row in rows:
                    # Ensure all required fields are present
                    prepared_row = {
                        "source": row.get("source"),
                        "name": row.get("name", ""),
                        "symbol": row.get("symbol"),
                        "date": row.get("date", ""),
                        "time": row.get("time"),
                        "price": row.get("price"),
                        "prev_close": row.get("prev_close"),
                        "change": row.get("change"),
                        "change_pct": row.get("change_pct"),
                        "raw_json": row.get("raw_json", ""),
                    }

                    # Validate required fields
                    if not prepared_row["name"] or not prepared_row["date"]:
                        logger.warning("Skipping row with missing name or date: %s", row)
                        continue

                    # Convert raw_json to string if it's a dict
                    if isinstance(prepared_row["raw_json"], dict):
                        prepared_row["raw_json"] = json.dumps(prepared_row["raw_json"], default=str)
                    elif prepared_row["raw_json"] is None:
                        prepared_row["raw_json"] = ""

                    prepared_rows.append(prepared_row)

                if not prepared_rows:
                    logger.warning("No valid rows to insert after validation")
                    return

                # Use executemany with named parameters
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO price_history
                    (source, name, symbol, date, time, price, prev_close, change, change_pct, raw_json)
                    VALUES (:source, :name, :symbol, :date, :time, :price, :prev_close, :change, :change_pct, :raw_json)
                    """,
                    prepared_rows,
                )
                inserted = cursor.rowcount
                conn.commit()
                logger.info("Bulk inserted %d rows (duplicates ignored)", inserted)

        except sqlite3.Error as e:
            logger.error("Error during bulk insert: %s", e)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("Unexpected error during bulk insert: %s", e)
            raise

