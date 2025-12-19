"""
PriceDataAgent collects financial market data from Yahoo Finance and FRED.

Collects indicators, saves snapshot JSON files, and maintains historical database.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import pytz
import yfinance as yf

from price_history_store import PriceHistoryStore

logger = logging.getLogger(__name__)

KST = pytz.timezone("Asia/Seoul")


def with_retry(fn: Callable[[], object], attempts: int = 3, delay: float = 0.5):
    """Retry a function call with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            logger.warning("Retry %d/%d after error: %s", i + 1, attempts, e)
            time.sleep(delay)


class PriceDataAgent:
    """Agent for collecting price data from Yahoo Finance and FRED."""

    def __init__(self, output_dir: Path | str = Path("./data")):
        """
        Initialize the PriceDataAgent.

        Args:
            output_dir: Directory for output files
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Yahoo Finance symbol dictionary
        self.yahoo_symbols = {
            # Major indicators
            "Dollar Index": "DX-Y.NYB",
            "US 10Y Treasury Yield": "^TNX",
            "US 30Y Treasury Yield": "^TYX",
            "US 2Y Treasury Yield": "^IRX",
            "Bitcoin": "BTC-USD",
            # Equity Indexes
            "S&P 500": "^GSPC",
            "Nasdaq Composite": "^IXIC",
            "Nasdaq 100": "^NDX",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
            "NYSE Composite": "^NYA",
            # Commodities
            "WTI Crude Oil": "CL=F",
            "Natural Gas": "NG=F",
            "Gold": "GC=F",
            "Silver": "SI=F",
            # FX
            "USD/KRW": "KRW=X",
            "USD/JPY": "JPY=X",
            "EUR/USD": "EUR=X",
        }

        # FRED series dictionary
        self.fred_series = {
            "Federal Funds Target Rate (Upper)": "DFEDTARU",
        }

        # Initialize history store
        self.history_store = PriceHistoryStore(self.output_dir / "price_history.db")

    def _fetch_yahoo_data(self, name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a single Yahoo Finance symbol.

        Args:
            name: Human-readable name
            symbol: Yahoo Finance symbol

        Returns:
            Dictionary with price data or None if fetch fails
        """
        try:
            ticker = yf.Ticker(symbol)
            fast_info = with_retry(lambda: ticker.fast_info)

            # Try to get last price and previous close
            last_price = None
            prev_close = None

            candidates_last = ["last_price", "lastPrice", "regular_market_price", "regularMarketPrice"]
            candidates_prev = [
                "previous_close",
                "previousClose",
                "regular_market_previous_close",
                "regularMarketPreviousClose",
            ]

            def pick(source, keys):
                for key in keys:
                    val = getattr(source, key, None)
                    if val is None and hasattr(source, "get"):
                        val = source.get(key)
                    if val is not None:
                        return val
                return None

            if fast_info:
                last_price = pick(fast_info, candidates_last)
                prev_close = pick(fast_info, candidates_prev)

            # Fallback to 1-minute history if fast_info is insufficient
            if last_price is None or prev_close is None:
                logger.warning("Fast info insufficient for %s (%s); using 1m history fallback", name, symbol)
                hist = with_retry(lambda: yf.download(symbol, period="5d", interval="1m", progress=False))
                if not hist.empty and "Close" in hist.columns:
                    closes = hist["Close"].dropna()
                    if len(closes) >= 2:
                        last_price = float(closes.iloc[-1])
                        prev_close = float(closes.iloc[-2])
                    elif len(closes) == 1:
                        last_price = float(closes.iloc[-1])
                        prev_close = float(closes.iloc[-1])

            if last_price is None or prev_close is None:
                logger.warning("Unable to get prices for %s (%s); skipping", name, symbol)
                return None

            # Calculate change
            change = None
            change_pct = 0.0
            try:
                if prev_close != 0:
                    change = float(last_price) - float(prev_close)
                    change_pct = (float(last_price) / float(prev_close) - 1.0) * 100.0
            except Exception:  # noqa: BLE001
                pass

            # Get timestamp
            ts_utc = pd.Timestamp.now(tz=pytz.UTC)
            as_kst = ts_utc.tz_convert(KST)

            result = {
                "name": name,
                "symbol": symbol,
                "date": as_kst.strftime("%Y-%m-%d"),
                "time": as_kst.strftime("%H:%M:%S"),
                "timestamp_utc": ts_utc.isoformat(),
                "timestamp_kst": as_kst.isoformat(),
                "price": float(last_price),
                "prev_close": float(prev_close),
                "change": change,
                "change_pct": change_pct,
            }

            time.sleep(0.15)  # Rate limiting
            return result

        except Exception as e:  # noqa: BLE001
            logger.error("Error fetching Yahoo data for %s (%s): %s", name, symbol, e)
            return None

    def _fetch_fred_data(self, name: str, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data for a single FRED series.

        Args:
            name: Human-readable name
            series_id: FRED series ID

        Returns:
            Dictionary with price data or None if fetch fails
        """
        try:
            # Use yfinance to fetch FRED data (yfinance supports some FRED series)
            # Alternatively, we can use pandas_datareader or fredapi
            # For now, try using yfinance with FRED: prefix
            symbol = f"FRED:{series_id}"
            ticker = yf.Ticker(symbol)
            hist = with_retry(lambda: ticker.history(period="5d", interval="1d", progress=False))

            if hist.empty or "Close" not in hist.columns:
                logger.warning("No FRED data available for %s (%s)", name, series_id)
                return None

            # Get the latest value
            closes = hist["Close"].dropna()
            if closes.empty:
                logger.warning("No close prices for FRED series %s (%s)", name, series_id)
                return None

            last_price = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_price

            # Calculate change
            change = None
            change_pct = 0.0
            try:
                if prev_close != 0:
                    change = float(last_price) - float(prev_close)
                    change_pct = (float(last_price) / float(prev_close) - 1.0) * 100.0
            except Exception:  # noqa: BLE001
                pass

            # Get timestamp
            ts_utc = pd.Timestamp.now(tz=pytz.UTC)
            as_kst = ts_utc.tz_convert(KST)

            result = {
                "name": name,
                "symbol": series_id,
                "date": as_kst.strftime("%Y-%m-%d"),
                "time": None,  # FRED data is typically daily, no time component
                "timestamp_utc": ts_utc.isoformat(),
                "timestamp_kst": as_kst.isoformat(),
                "price": last_price,
                "prev_close": prev_close,
                "change": change,
                "change_pct": change_pct,
            }

            time.sleep(0.15)  # Rate limiting
            return result

        except Exception as e:  # noqa: BLE001
            logger.error("Error fetching FRED data for %s (%s): %s", name, series_id, e)
            # Try alternative method: use pandas_datareader if available
            try:
                import pandas_datareader.data as web

                data = web.DataReader(series_id, "fred", start=pd.Timestamp.now() - pd.Timedelta(days=5))
                if not data.empty:
                    last_price = float(data.iloc[-1, 0])
                    prev_close = float(data.iloc[-2, 0]) if len(data) >= 2 else last_price

                    ts_utc = pd.Timestamp.now(tz=pytz.UTC)
                    as_kst = ts_utc.tz_convert(KST)

                    change = None
                    change_pct = 0.0
                    try:
                        if prev_close != 0:
                            change = float(last_price) - float(prev_close)
                            change_pct = (float(last_price) / float(prev_close) - 1.0) * 100.0
                    except Exception:  # noqa: BLE001
                        pass

                    return {
                        "name": name,
                        "symbol": series_id,
                        "date": as_kst.strftime("%Y-%m-%d"),
                        "time": None,
                        "timestamp_utc": ts_utc.isoformat(),
                        "timestamp_kst": as_kst.isoformat(),
                        "price": last_price,
                        "prev_close": prev_close,
                        "change": change,
                        "change_pct": change_pct,
                    }
            except ImportError:
                logger.warning("pandas_datareader not available for FRED data")
            except Exception as e2:  # noqa: BLE001
                logger.error("Alternative FRED fetch also failed: %s", e2)

            return None

    def collect_all_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect all data from Yahoo Finance and FRED.

        Returns:
            Dictionary with 'yahoo_finance' and 'fred' keys containing lists of data entries
        """
        logger.info("Starting data collection for all indicators")

        yahoo_results: List[Dict[str, Any]] = []
        fred_results: List[Dict[str, Any]] = []

        # Collect Yahoo Finance data
        logger.info("Collecting Yahoo Finance data for %d symbols", len(self.yahoo_symbols))
        for name, symbol in self.yahoo_symbols.items():
            logger.debug("Fetching %s (%s)", name, symbol)
            data = self._fetch_yahoo_data(name, symbol)
            if data:
                yahoo_results.append(data)
            else:
                logger.warning("Failed to fetch data for %s (%s)", name, symbol)

        # Collect FRED data
        logger.info("Collecting FRED data for %d series", len(self.fred_series))
        for name, series_id in self.fred_series.items():
            logger.debug("Fetching %s (%s)", name, series_id)
            data = self._fetch_fred_data(name, series_id)
            if data:
                fred_results.append(data)
            else:
                logger.warning("Failed to fetch data for %s (%s)", name, series_id)

        result = {
            "yahoo_finance": yahoo_results,
            "fred": fred_results,
        }

        logger.info(
            "Collection complete: %d Yahoo Finance entries, %d FRED entries",
            len(yahoo_results),
            len(fred_results),
        )

        return result

    def save_data(self, data: Dict[str, List[Dict[str, Any]]]) -> Path:
        """
        Save collected data to a JSON snapshot file.

        Args:
            data: Collected data dictionary

        Returns:
            Path to the saved file
        """
        ts_utc = pd.Timestamp.now(tz=pytz.UTC)
        as_kst = ts_utc.tz_convert(KST)
        filename = f"snapshot_{as_kst.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        snapshot = {
            "timestamp_utc": ts_utc.isoformat(),
            "timestamp_kst": as_kst.isoformat(),
            "data": data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.info("Saved snapshot data to %s", filepath)
        return filepath

    def save_history(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Save collected data to the historical database.

        Args:
            data: Collected data dictionary
        """
        self.history_store.upsert_from_collection(data)

    def run_daily_collection(self) -> Path:
        """
        Run the daily data collection process.

        Collects all data, saves snapshot JSON, and updates historical database.

        Returns:
            Path to the saved snapshot file
        """
        logger.info("Starting daily collection")
        data = self.collect_all_data()
        filepath = self.save_data(data)
        self.save_history(data)
        logger.info("Daily collection complete")
        return filepath


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    agent = PriceDataAgent()
    agent.run_daily_collection()

