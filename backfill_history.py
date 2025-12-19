"""
Backfill script to load full historical data from Yahoo Finance.

Downloads historical data for all symbols in PriceDataAgent.yahoo_symbols
and stores them in the SQLite database using bulk insert.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

from price_data_agent import PriceDataAgent
from price_history_store import PriceHistoryStore

logger = logging.getLogger(__name__)


def with_retry(fn, attempts: int = 3, delay: float = 1.0):
    """Retry a function call with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            logger.warning("Retry %d/%d after error: %s", i + 1, attempts, e)
            time.sleep(delay * (i + 1))  # Exponential backoff


def fetch_full_history_yahoo(name: str, symbol: str) -> List[Dict[str, Any]]:
    """
    Fetch full historical data for a Yahoo Finance symbol using the chart API.

    Args:
        name: Human-readable name
        symbol: Yahoo Finance symbol

    Returns:
        List of row dictionaries ready for insert_rows
    """
    logger.info("Fetching full history for %s (%s)", name, symbol)

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "range": "max",
        "interval": "1d",
        "includePrePost": "false",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        def fetch():
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                logger.warning(
                    "HTTP %d for %s (%s): %s", response.status_code, name, symbol, response.text[:200]
                )
                return None
            return response.json()

        data = with_retry(fetch, attempts=3, delay=1.0)

        if data is None:
            logger.warning("No data returned for %s (%s)", name, symbol)
            return []

        # Parse JSON response
        result_list = data.get("chart", {}).get("result", [])
        if not result_list:
            logger.warning("Empty result list for %s (%s)", name, symbol)
            return []

        result = result_list[0]
        timestamps = result.get("timestamp", []) or []
        indicators = result.get("indicators", {})
        quote_list = indicators.get("quote", [])

        if not quote_list:
            logger.warning("No quote data for %s (%s)", name, symbol)
            return []

        quote = quote_list[0]
        closes = quote.get("close", []) or []

        if not timestamps or not closes:
            logger.warning("Empty timestamps or closes for %s (%s)", name, symbol)
            return []

        # Build rows
        rows = []
        prev_close = None

        for i, ts in enumerate(timestamps):
            if i >= len(closes):
                break

            price = closes[i]
            if price is None:
                continue

            # Convert timestamp to datetime
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            # Calculate change
            change = None
            change_pct = None
            if prev_close is not None and prev_close not in (0, None):
                change = price - prev_close
                change_pct = (change / prev_close) * 100.0

            row = {
                "source": "yahoo_finance",
                "name": name,
                "symbol": symbol,
                "date": date_str,
                "time": None,
                "price": float(price),
                "prev_close": float(prev_close) if prev_close is not None else None,
                "change": float(change) if change is not None else None,
                "change_pct": float(change_pct) if change_pct is not None else None,
                "raw_json": json.dumps(
                    {
                        "symbol": symbol,
                        "timestamp": ts,
                        "date": date_str,
                        "close": price,
                        "index": i,
                    },
                    ensure_ascii=False,
                ),
            }

            rows.append(row)
            prev_close = price

        if len(rows) == 0:
            logger.warning("No valid rows generated for %s (%s)", name, symbol)

        return rows

    except Exception as e:  # noqa: BLE001
        logger.exception("Error fetching full history for %s (%s): %s", name, symbol, e)
        return []


def backfill_all_symbols(
    agent: PriceDataAgent,
    history_store: PriceHistoryStore,
    batch_size: int = 1000,
) -> None:
    """
    Backfill historical data for all symbols in PriceDataAgent.

    Args:
        agent: PriceDataAgent instance
        history_store: PriceHistoryStore instance
        batch_size: Number of rows to insert per batch (default: 1000)
    """
    logger.info("Starting backfill for %d Yahoo Finance symbols", len(agent.yahoo_symbols))

    total_rows_inserted = 0
    total_symbols_processed = 0
    failed_symbols = []

    for name, symbol in agent.yahoo_symbols.items():
        logger.info("Backfilling Yahoo history for %s (%s)", name, symbol)
        try:
            rows = fetch_full_history_yahoo(name, symbol)
            logger.info("Fetched %d rows for %s", len(rows), name)

            if not rows:
                logger.warning("Skipping %s (%s) - no rows fetched", name, symbol)
                failed_symbols.append((name, symbol, "No rows"))
                continue

            # Insert in batches
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                history_store.insert_rows(batch)
                total_rows_inserted += len(batch)
                logger.debug(
                    "Inserted batch %d-%d for %s (%s)", i + 1, min(i + batch_size, len(rows)), name, symbol
                )

            total_symbols_processed += 1
            logger.info("Inserted %d rows for %s", len(rows), name)

            # Rate limiting between symbols
            time.sleep(0.5)

        except Exception as e:  # noqa: BLE001
            logger.exception("Error backfilling %s (%s): %s", name, symbol, e)
            failed_symbols.append((name, symbol, str(e)))

    logger.info("=" * 60)
    logger.info("Backfill complete!")
    logger.info("Total symbols processed: %d", total_symbols_processed)
    logger.info("Total rows inserted: %d", total_rows_inserted)
    if failed_symbols:
        logger.warning("Failed symbols (%d):", len(failed_symbols))
        for name, symbol, reason in failed_symbols:
            logger.warning("  - %s (%s): %s", name, symbol, reason)
    logger.info("=" * 60)


def main():
    """Main entry point for backfill script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize agent and history store
    output_dir = Path("./data")
    agent = PriceDataAgent(output_dir=output_dir)
    history_store = PriceHistoryStore(output_dir / "price_history.db")

    # Run backfill
    logger.info("Starting historical data backfill")
    backfill_all_symbols(agent, history_store, batch_size=1000)
    logger.info("Backfill script completed")


if __name__ == "__main__":
    main()

