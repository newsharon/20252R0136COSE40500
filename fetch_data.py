"""
Quick MVP pipeline to fetch market data for selected tickers using yfinance.

Outputs:
- ./data/eod.parquet (daily OHLCV, 2 years)
- ./data/intraday_1m.parquet (1-minute OHLCV, last 5 days)
- ./data/snapshot_latest.csv (latest close and percent change)
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pytz
import yfinance as yf


DATA_DIR = Path("./data")
EOD_PATH = DATA_DIR / "eod.parquet"
INTRADAY_PATH = DATA_DIR / "intraday_1m.parquet"
SNAPSHOT_PATH = DATA_DIR / "snapshot_latest.csv"

KST = pytz.timezone("Asia/Seoul")
SNAPSHOT_COLS = ["ticker", "ts", "as_of_kst", "close", "change_pct"]

DEFAULT_TICKERS = (
    "NVDA",
    "MSFT",
    "TSLA",
    "LLY",
    "BAC",
    "KO",
    "^GSPC",
    "^IXIC",
    "^NDX",
    "^DJI",
    "^RUT",
    "^NYA",
    "DX-Y.NYB",
    "^TNX",
    "^TYX",
    "CL=F",
    "NG=F",
    "BTC-USD",
)


def tidy_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "ticker", "open", "high", "low", "close", "volume"])

    if not isinstance(df.columns, pd.MultiIndex):
        fields = [c for c in df.columns if str(c).lower() in {"open", "high", "low", "close", "adj close", "volume"}]
        if fields:
            df = pd.concat({"UNKNOWN": df[fields]}, axis=1)
        else:
            return pd.DataFrame(columns=["ts", "ticker", "open", "high", "low", "close", "volume"])

    df = df.copy()
    raw_cols = [(str(a), str(b)) for a, b in df.columns]
    field_names = {"open", "high", "low", "close", "adj close", "volume"}
    if raw_cols:
        first = raw_cols[0]
        first_lower = (first[0].lower(), first[1].lower())
        if first_lower[0] in field_names and first_lower[1] not in field_names:
            ticker_idx, field_idx = 1, 0
        else:
            ticker_idx, field_idx = 0, 1
    else:
        ticker_idx, field_idx = 0, 1

    new_cols = [(col[ticker_idx], col[field_idx]) for col in raw_cols]
    df.columns = pd.MultiIndex.from_tuples(new_cols, names=["ticker", "field"])
    df = df.sort_index(axis=1)

    df = df.stack(level=0)
    df.index = df.index.set_names(["ts", "ticker"])
    df = df.reset_index()

    df = df.rename(columns=lambda c: c.lower() if isinstance(c, str) else c)
    rename_map = {
        "adj close": "adj close",
        "close": "close",
        "high": "high",
        "low": "low",
        "open": "open",
        "volume": "volume",
    }
    df = df.rename(columns=rename_map)

    keep = ["ts", "ticker", "open", "high", "low", "close", "volume"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA

    drop_these = [c for c in ["adj close", "dividends", "stock splits"] if c in df.columns]
    if drop_these:
        df = df.drop(columns=drop_these, errors="ignore")

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.reindex(columns=keep)
    df["ticker"] = df["ticker"].astype(str)
    bad = {"open", "high", "low", "close", "adj close", "volume"}
    df = df[~df["ticker"].str.lower().isin(bad)]

    if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        df = df.dropna(subset=["close"], how="all")

    return df


def with_retry(fn: Callable[[], object], attempts: int = 3, delay: float = 0.5):
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)


def ensure_data_dir(path: Path) -> None:
    os.makedirs(path, exist_ok=True)


def fetch_eod(tickers: Iterable[str]) -> pd.DataFrame:
    logging.info("Fetching daily OHLCV for %s", ", ".join(tickers))
    df = yf.download(
        tickers=list(tickers),
        period="2y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )
    tidy = tidy_ohlcv(df)
    if tidy.empty:
        raise RuntimeError("No daily data returned from yfinance.")

    tickers_upper = {t.upper() for t in tickers}
    tidy = tidy[tidy["ticker"].str.upper().isin(tickers_upper)]
    return tidy.reset_index(drop=True)


def fetch_intraday_1m(tickers: Iterable[str]) -> pd.DataFrame:
    logging.info("Fetching 1-minute OHLCV for %s", ", ".join(tickers))
    df = yf.download(
        tickers=list(tickers),
        period="5d",
        interval="1m",
        auto_adjust=False,
        group_by="ticker",
        threads=False,
        progress=False,
    )
    tidy = tidy_ohlcv(df)
    if tidy.empty:
        raise RuntimeError("No intraday data returned from yfinance.")

    tickers_upper = {t.upper() for t in tickers}
    tidy = tidy[tidy["ticker"].str.upper().isin(tickers_upper)]
    return tidy.reset_index(drop=True)


def fetch_snapshot(tickers: Iterable[str]) -> pd.DataFrame:
    rows = []
    logging.info("Creating latest snapshot for %s", ", ".join(tickers))

    def get_fast_info_prices(ticker: str):
        fi = with_retry(lambda: yf.Ticker(ticker).fast_info)
        last_candidates = [
            "last_price",
            "lastPrice",
            "regular_market_price",
            "regularMarketPrice",
        ]
        prev_candidates = [
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

        if fi is None:
            return None, None

        last_val = pick(fi, last_candidates)
        prev_val = pick(fi, prev_candidates)
        return last_val, prev_val

    for ticker in tickers:
        last_price, prev_close = get_fast_info_prices(ticker)

        if last_price is None or prev_close is None:
            logging.warning("Snapshot: fast_info insufficient for %s; using 1m history fallback", ticker)
            hist = with_retry(lambda: yf.download(ticker, period="5d", interval="1m", progress=False))
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) >= 2:
                    last_price = float(closes.iloc[-1])
                    prev_close = float(closes.iloc[-2])
                elif len(closes) == 1:
                    last_price = float(closes.iloc[-1])
                    prev_close = float(closes.iloc[-1])

        if last_price is None or prev_close is None:
            logging.warning("Snapshot: unable to get prices for %s; skipping", ticker)
            continue

        change_pct = 0.0
        try:
            if prev_close != 0:
                change_pct = (float(last_price) / float(prev_close) - 1.0) * 100.0
        except Exception:  # noqa: BLE001
            pass

        ts_utc = pd.Timestamp.now(tz=pytz.UTC)
        as_kst = ts_utc.tz_convert(KST)

        rows.append(
            {
                "ticker": ticker,
                "ts": ts_utc.isoformat(),
                "as_of_kst": as_kst.isoformat(),
                "close": float(last_price),
                "change_pct": float(change_pct),
            }
        )
        time.sleep(0.15)

    snapshot_df = pd.DataFrame(rows)
    snapshot_df = snapshot_df.reindex(columns=SNAPSHOT_COLS)
    assert set(snapshot_df.columns) == set(SNAPSHOT_COLS), "snapshot columns mismatch"
    assert snapshot_df["ticker"].notna().all()
    assert snapshot_df["close"].notna().all()
    snapshot_df.to_csv(SNAPSHOT_PATH, index=False)
    logging.info("Saved snapshot data to %s", SNAPSHOT_PATH)
    return snapshot_df


def main(tickers: Iterable[str] = DEFAULT_TICKERS) -> None:
    ensure_data_dir(DATA_DIR)

    eod_df = fetch_eod(tickers)
    eod_df.to_parquet(EOD_PATH, engine="pyarrow", index=False)
    logging.info("Saved daily data to %s", EOD_PATH)

    intraday_df = fetch_intraday_1m(tickers)
    intraday_df.to_parquet(INTRADAY_PATH, engine="pyarrow", index=False)
    logging.info("Saved intraday data to %s", INTRADAY_PATH)

    fetch_snapshot(tickers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

