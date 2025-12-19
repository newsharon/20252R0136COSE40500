"""Minimal smoke test for analyzers."""

from __future__ import annotations

import pandas as pd

import chart_tool as ct
from agents.base import AgentResult
from agents.macro import MacroAnalyzer
from agents.sector import SectorAnalyzer
from agents.ticker import TickerAnalyzer


def main() -> None:
    snap = ct.load_snapshot()
    eod = ct.load_eod()
    iad = ct.load_intraday_1m()

    macro = MacroAnalyzer()
    sector = SectorAnalyzer()
    ticker = TickerAnalyzer(tickers=["NVDA", "MSFT", "TSLA", "LLY", "BAC", "KO"])

    results: list[AgentResult] = []
    results.append(macro.analyze({"snapshot": snap, "eod": eod}))
    results.append(sector.analyze({"snapshot": snap, "eod": eod}))
    results.append(ticker.analyze({"snapshot": snap, "eod": eod, "intraday": iad}))

    for r in results:
        print(f"[{r.section}] {type(r).__name__}")
        for b in r.bullets:
            print(" -", b)
        print(" summary:", r.summary)
        print(" evidence:", r.evidence)
        print()


if __name__ == "__main__":
    main()


