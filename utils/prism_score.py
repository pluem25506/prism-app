"""
PRISM Backtest Engine -- simulation harness only.

As of this rewrite, the actual 9-layer scoring is no longer a separate
parallel implementation here -- it calls backend.prism_score.compute_prism_score
(ticker, year), the exact same scorer the live Analyze page uses, just
pointed at a historical fiscal year instead of today. This module now only
owns what's specific to backtesting: fetching a long price history, running
the year-by-year scoring loop, simulating the BUY/HOLD/SELL strategy at each
rebalance date, building the equity curve, and computing summary stats.

L7 Portfolio Fit is always unavailable here (it depends on the user's
*current* holdings, which has no historical meaning) -- backend.prism_score
already returns S7=None for any explicit year, and the composite score
renormalizes around it automatically, same as any other missing layer.
"""

import pandas as pd
import streamlit as st

from backend.data_fetcher import fetch_price_history_long
from backend.prism_score import WEIGHTS, compute_prism_score

DEFAULT_RISK_FREE_RATE = 0.04

LAYER_LABELS = {
    "S1": "Quality Screen", "S2": "Value Creation", "S3": "Valuation",
    "S4": "Risk & Leverage", "S5": "Monte Carlo", "S6": "Time-Series Momentum",
    "S7": "Portfolio Fit (N/A historically)", "S8": "Operating & Financial Leverage",
    "S9": "Contextual Intelligence",
}


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period_years) -> pd.DataFrame:
    """Daily OHLCV for `ticker`. `period_years` is an int, or "max" for the
    full available history."""
    period = "max" if period_years == "max" else f"{int(period_years)}y"
    df = fetch_price_history_long(ticker, period=period)
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' not found on yfinance, or has no price history.")
    return df


@st.cache_data(ttl=3600)
def _cached_score_for_year(ticker: str, year: int) -> dict:
    return compute_prism_score(ticker, year)


def score_to_signal(score: float, buy_threshold: float, sell_threshold: float) -> str:
    if score >= buy_threshold:
        return "BUY"
    if score < sell_threshold:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

_REBALANCE_MONTHS = {
    "annual": [1],
    "semiannual": [1, 7],
    "quarterly": [1, 4, 7, 10],
}


def _rebalance_dates(daily_prices: pd.DataFrame, rebalance: str):
    """First trading day on/after the 1st of each rebalance month, for every
    year the ticker has price data."""
    months = _REBALANCE_MONTHS[rebalance]
    years = sorted(set(daily_prices.index.year))
    dates = []
    for year in years:
        for month in months:
            month_rows = daily_prices[(daily_prices.index.year == year) & (daily_prices.index.month == month)]
            if not month_rows.empty:
                dates.append(month_rows.index[0])
    return dates


def run_backtest(
    ticker: str,
    benchmark: str,
    period_years,
    rebalance: str,
    buy_threshold: float,
    sell_threshold: float,
    risk_free_when_out: bool = False,
) -> dict:
    daily_prices = fetch_stock_data(ticker, period_years)
    benchmark_prices = fetch_stock_data(benchmark, period_years)

    # A target ticker's own history can predate the benchmark's existence
    # (e.g. MSFT back to 1986 vs QQQ's 1999 inception, or a 10y lookback on
    # a ticker with only 3y of real history). Reindexing the benchmark onto
    # dates before it existed leaves leading NaNs with nothing to
    # forward-fill from, which poisons every benchmark stat. Clip to the
    # actual overlap instead.
    common_start = max(daily_prices.index[0], benchmark_prices.index[0])
    common_end = min(daily_prices.index[-1], benchmark_prices.index[-1])
    if common_start >= common_end:
        raise ValueError(f"'{ticker}' and benchmark '{benchmark}' have no overlapping price history.")
    daily_prices = daily_prices[(daily_prices.index >= common_start) & (daily_prices.index <= common_end)]

    start_year = int(daily_prices.index[0].year)
    end_year = int(daily_prices.index[-1].year)

    rows = []
    for year in range(start_year, end_year + 1):
        result = _cached_score_for_year(ticker, year)
        score = result["prism_score"]
        signal = score_to_signal(score, buy_threshold, sell_threshold)
        row = {
            "year": year, "score": score, "signal": signal,
            "data_quality": "full" if result["available_layers"] >= 7 else ("partial" if result["available_layers"] >= 4 else "minimal"),
            "years_available": result["available_layers"],
        }
        row.update(result["sub_scores"])
        rows.append(row)
    scores_df = pd.DataFrame(rows).set_index("year")

    # --- Simulate the strategy at each rebalance date ---
    rebal_dates = _rebalance_dates(daily_prices, rebalance)
    trade_log = []
    invested = False
    entry_price = None

    for date in rebal_dates:
        year = date.year
        if year not in scores_df.index:
            continue
        score = scores_df.loc[year, "score"]
        signal = scores_df.loc[year, "signal"]
        price_at_date = float(daily_prices.loc[daily_prices.index <= date, "Close"].iloc[-1])

        if signal == "BUY" and not invested:
            invested = True
            entry_price = price_at_date
            trade_log.append({"date": date, "action": "BUY", "price": price_at_date, "score": score})
        elif signal == "SELL" and invested:
            pnl_pct = (price_at_date / entry_price - 1) * 100
            trade_log.append({"date": date, "action": "SELL", "price": price_at_date, "score": score, "pnl_pct": pnl_pct})
            invested = False
            entry_price = None

    # --- Equity curves ---
    idx = daily_prices.index
    equity = pd.Series(index=idx, dtype=float)
    equity_val = 100.0
    is_invested = False
    entry_px = None
    entry_equity = None
    trade_iter = iter(trade_log)
    next_trade = next(trade_iter, None)
    risk_free_daily = DEFAULT_RISK_FREE_RATE / 252

    for date in idx:
        while next_trade is not None and next_trade["date"] <= date:
            if next_trade["action"] == "BUY":
                is_invested = True
                entry_px = next_trade["price"]
                entry_equity = equity_val
            elif next_trade["action"] == "SELL":
                equity_val = entry_equity * (float(daily_prices.loc[date, "Close"]) / entry_px) if entry_px else equity_val
                is_invested = False
                entry_px = None
            next_trade = next(trade_iter, None)

        if is_invested and entry_px:
            equity.loc[date] = entry_equity * (float(daily_prices.loc[date, "Close"]) / entry_px)
        else:
            if risk_free_when_out and not equity.empty and date != idx[0]:
                prev = equity.iloc[equity.index.get_loc(date) - 1] if equity.index.get_loc(date) > 0 else equity_val
                equity.loc[date] = (prev if pd.notna(prev) else equity_val) * (1 + risk_free_daily)
            else:
                equity.loc[date] = equity_val

    equity = equity.ffill().fillna(100.0)

    benchmark_aligned = benchmark_prices["Close"].reindex(idx, method="ffill")
    benchmark_curve = (benchmark_aligned / benchmark_aligned.iloc[0]) * 100

    stats = compute_stats(equity, benchmark_curve, trade_log)

    return {
        "scores": scores_df, "daily_prices": daily_prices, "benchmark_prices": benchmark_prices,
        "trade_log": trade_log, "equity_curve": equity, "benchmark_curve": benchmark_curve,
        "stats": stats, "ticker": ticker, "benchmark": benchmark,
    }


def compute_stats(equity: pd.Series, benchmark: pd.Series, trade_log: list) -> dict:
    def series_stats(curve: pd.Series):
        total_return = (curve.iloc[-1] / curve.iloc[0] - 1) * 100
        n_years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1 / 365.25)
        cagr = ((curve.iloc[-1] / curve.iloc[0]) ** (1 / n_years) - 1) * 100
        daily_ret = curve.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std() * (252 ** 0.5)) if daily_ret.std() else 0.0
        rolling_max = curve.expanding().max()
        drawdown = (curve - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        return total_return, cagr, sharpe, max_dd

    strat_total, strat_cagr, strat_sharpe, strat_dd = series_stats(equity)
    bench_total, bench_cagr, bench_sharpe, bench_dd = series_stats(benchmark)

    sells = [t for t in trade_log if t["action"] == "SELL"]
    wins = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)
    win_rate = (wins / len(sells) * 100) if sells else None

    return {
        "strategy": {"total_return": strat_total, "cagr": strat_cagr, "sharpe": strat_sharpe,
                     "max_drawdown": strat_dd, "win_rate": win_rate, "n_trades": len(sells)},
        "benchmark": {"total_return": bench_total, "cagr": bench_cagr, "sharpe": bench_sharpe,
                      "max_drawdown": bench_dd, "win_rate": None, "n_trades": 1},
        "alpha": strat_cagr - bench_cagr,
    }
