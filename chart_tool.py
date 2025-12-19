"""Utility readers for locally stored market data.

Loads the outputs produced by `fetch_data.py` and provides simple filters by
ticker and time window. Designed for zero network usage and pandas-only
workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


DATA_DIR = Path("./data")
EOD_PATH = DATA_DIR / "eod.parquet"
INTRADAY_PATH = DATA_DIR / "intraday_1m.parquet"
SNAPSHOT_PATH = DATA_DIR / "snapshot_latest.csv"


def ensure_schema(df: pd.DataFrame, required_cols: Sequence[str], name: str) -> None:
    """Validate that the dataframe contains all required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing expected columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _normalize_bounds(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _filter_timeframe(
    df: pd.DataFrame, start: str | None, end: str | None
) -> pd.DataFrame:
    start_ts = _normalize_bounds(start)
    end_ts = _normalize_bounds(end)
    if start_ts is not None:
        df = df[df["ts"] >= start_ts]
    if end_ts is not None:
        df = df[df["ts"] <= end_ts]
    return df


def _filter_tickers(df: pd.DataFrame, tickers: Iterable[str] | None) -> pd.DataFrame:
    if not tickers:
        return df
    tickers_set = {t.upper() for t in tickers}
    return df[df["ticker"].str.upper().isin(tickers_set)]


def load_snapshot() -> pd.DataFrame:
    """Load the latest snapshot CSV into a dataframe."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Snapshot file not found at {SNAPSHOT_PATH}")

    df = pd.read_csv(SNAPSHOT_PATH)
    required = ["ticker", "ts", "close", "change_pct"]
    ensure_schema(df, required_cols=required, name="snapshot")

    if "as_of_kst" not in df.columns:
        df = df.assign(as_of_kst=pd.NA)

    df = df[["ticker", "ts", "as_of_kst", "close", "change_pct"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="raise")
    return df


def load_eod(
    tickers: List[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load daily OHLCV data, optionally filtered by ticker and time window."""
    if not EOD_PATH.exists():
        raise FileNotFoundError(f"EOD parquet not found at {EOD_PATH}")

    df = pd.read_parquet(EOD_PATH)
    required = ["ts", "ticker", "open", "high", "low", "close", "volume"]
    ensure_schema(df, required_cols=required, name="eod")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="raise")

    df = _filter_tickers(df, tickers)
    df = _filter_timeframe(df, start=start, end=end)
    return df.reset_index(drop=True)


def load_intraday_1m(
    tickers: List[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load 1-minute OHLCV data, optionally filtered by ticker and time window."""
    if not INTRADAY_PATH.exists():
        raise FileNotFoundError(f"Intraday parquet not found at {INTRADAY_PATH}")

    df = pd.read_parquet(INTRADAY_PATH)
    required = ["ts", "ticker", "open", "high", "low", "close", "volume"]
    ensure_schema(df, required_cols=required, name="intraday_1m")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="raise")

    df = _filter_tickers(df, tickers)
    df = _filter_timeframe(df, start=start, end=end)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("=== Snapshot ===")
    try:
        snap = load_snapshot()
        print("shape:", snap.shape)
        print(snap.head(3))
    except Exception as exc:  # noqa: BLE001
        print("snapshot unavailable:", exc)

    print("\n=== EOD (all) ===")
    try:
        eod_all = load_eod()
        print("shape:", eod_all.shape)
        print(eod_all.head(3))
        eod_filt = load_eod(tickers=["NVDA"], start="2024-01-01", end="2024-12-31")
        print("filtered shape:", eod_filt.shape)
        print(eod_filt.head(3))
    except Exception as exc:  # noqa: BLE001
        print("eod unavailable:", exc)

    print("\n=== Intraday (all) ===")
    try:
        intraday_all = load_intraday_1m()
        print("shape:", intraday_all.shape)
        print(intraday_all.head(3))
        intraday_filt = load_intraday_1m(
            tickers=["NVDA"], start="2024-11-01 00:00", end="2024-11-05 23:59"
        )
        print("filtered shape:", intraday_filt.shape)
        print(intraday_filt.head(3))
    except Exception as exc:  # noqa: BLE001
        print("intraday unavailable:", exc)

