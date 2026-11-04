"""
PRISM Backtest Engine -- scoring + simulation core.

Independent of backend/ (the live 9-layer analyzer): this module scores a
ticker YEAR BY YEAR from annual fundamentals + daily price data, for the
standalone Backtest page. Layer names/weights are specific to this engine
and are not the same L1-L9 as the live app.

Known data limitation: yfinance's free annual financials/balance_sheet/
cashflow only go back ~4 fiscal years, even when a 10-20 year price history
is available. Years outside that window simply have no fundamentals --
compute_prism_score() degrades gracefully for those years (skips the
layers that need them and renormalizes the rest) rather than crashing or
guessing.
"""

import functools
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

DEFAULT_RISK_FREE_RATE = 0.04

LAYER_WEIGHTS = {
    "L1_valuation": 0.15,
    "L2_profitability": 0.15,
    "L3_growth": 0.15,
    "L4_safety": 0.10,
    "L5_efficiency": 0.10,
    "L6_dividends": 0.05,
    "L7_momentum": 0.15,
    "L8_esg": 0.05,
    "L9_relative_strength": 0.10,
}

# Bitcoin and other non-equity tickers have no fundamentals at all -- rescale
# onto the two layers that still make sense (price momentum + relative
# strength) rather than reporting a near-zero score dragged down by five
# permanently-missing layers.
NO_FUNDAMENTALS_WEIGHTS = {
    "L7_momentum": 0.60,
    "L9_relative_strength": 0.40,
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period_years) -> pd.DataFrame:
    """Daily OHLCV for `ticker`. `period_years` is an int, or "max" for the
    full available history."""
    period = "max" if period_years == "max" else f"{int(period_years)}y"
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval="1d")
    if df is None or df.empty:
        raise ValueError(f"Ticker '{ticker}' not found on yfinance, or has no price history.")
    # Crypto (BTC-USD) trades in UTC while equities trade in America/New_York
    # -- reindex()/comparisons between a crypto series and an equity
    # benchmark would otherwise compare absolute instants across two
    # different tz offsets and barely ever match. Drop to calendar-date-only,
    # tz-naive, so every series in the engine lines up on the same axis.
    df.index = df.index.tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])


def _item_row(df, keywords):
    """Fuzzy row lookup in a financials/balance_sheet/cashflow DataFrame."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    normalized_keywords = [kw.lower().replace(" ", "") for kw in keywords]
    for row_label in df.index:
        row_norm = str(row_label).lower().replace(" ", "")
        for kw in normalized_keywords:
            if kw in row_norm or row_norm in kw:
                return df.loc[row_label]
    return None


@st.cache_data(ttl=3600)
def fetch_fundamentals_annual(ticker: str) -> dict:
    """Annual fundamentals keyed by fiscal year (int). See module docstring
    for the ~4-year yfinance limitation; missing years simply aren't keys
    in the returned dict."""
    tk = yf.Ticker(ticker)

    def safe(attr):
        try:
            return getattr(tk, attr)
        except Exception:
            return None

    financials = safe("financials")
    balance_sheet = safe("balance_sheet")
    cashflow = safe("cashflow")
    info = safe("info") or {}

    rows = {
        "revenue": _item_row(financials, ["Total Revenue", "TotalRevenue"]),
        "net_income": _item_row(financials, ["Net Income", "NetIncome", "Net Income Common Stockholders"]),
        "ebitda": _item_row(financials, ["EBITDA", "Normalized EBITDA"]),
        "ebit": _item_row(financials, ["EBIT", "Operating Income"]),
        "interest_expense": _item_row(financials, ["Interest Expense"]),
        "cost_of_revenue": _item_row(financials, ["Cost Of Revenue", "Cost Of Goods Sold", "Reconciled Cost Of Revenue"]),
        "total_assets": _item_row(balance_sheet, ["Total Assets"]),
        "total_equity": _item_row(balance_sheet, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"]),
        "total_debt": _item_row(balance_sheet, ["Total Debt"]),
        "cash": _item_row(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        "current_assets": _item_row(balance_sheet, ["Total Current Assets", "Current Assets"]),
        "current_liabilities": _item_row(balance_sheet, ["Total Current Liabilities", "Current Liabilities"]),
        "inventory": _item_row(balance_sheet, ["Inventory"]),
        "operating_cashflow": _item_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]),
    }

    years = set()
    for row in rows.values():
        if row is not None:
            years.update(c.year for c in row.index)

    shares_out = info.get("sharesOutstanding")

    data = {}
    for year in years:
        def val(row, _year=year):
            if row is None:
                return None
            for c in row.index:
                if c.year == _year:
                    v = row[c]
                    return float(v) if pd.notna(v) else None
            return None

        data[year] = {key: val(row) for key, row in rows.items()}
        data[year]["shares_outstanding"] = shares_out

    return data


@st.cache_data(ttl=3600)
def fetch_dividends(ticker: str) -> pd.Series:
    try:
        return yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def fetch_sustainability_score(ticker: str):
    """Best-effort ESG score 0-100 (higher=better), or None if unavailable.

    Yahoo deprecated the underlying Sustainalytics endpoint -- confirmed
    empty for essentially every ticker as of late 2026 -- so this will
    almost always fall through to None, matching the Part 4 spec's "do NOT
    penalize" handling in compute_prism_score.
    """
    try:
        sus = yf.Ticker(ticker).sustainability
        if sus is None or sus.empty or "totalEsg" not in sus.index:
            return None
        total_esg = float(sus.loc["totalEsg", "Value"])
        # Sustainalytics risk score: 0 (best) - 40+ (worst) -> 100-0 scale.
        return max(0.0, min(100.0, 100 - total_esg * 2.5))
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_macro_data(start_year: int, end_year: int) -> dict:
    """{year: {"risk_free_rate": float, "unemployment": float}}. Falls back
    to a flat 4% risk-free rate for every year if FRED is unavailable --
    never raises."""
    fred_key = None
    try:
        fred_key = st.secrets.get("FRED_API_KEY")
    except Exception:
        pass

    fallback = {y: {"risk_free_rate": DEFAULT_RISK_FREE_RATE, "unemployment": None} for y in range(start_year, end_year + 1)}

    if not fred_key or not FRED_AVAILABLE:
        return fallback

    try:
        fred = Fred(api_key=fred_key)
        dgs10 = fred.get_series("DGS10").dropna()
        unrate = fred.get_series("UNRATE").dropna()
    except Exception:
        return fallback

    result = {}
    for year in range(start_year, end_year + 1):
        try:
            year_rate = dgs10[dgs10.index.year == year]
            rf = float(year_rate.mean()) / 100 if not year_rate.empty else DEFAULT_RISK_FREE_RATE
        except Exception:
            rf = DEFAULT_RISK_FREE_RATE
        try:
            year_unrate = unrate[unrate.index.year == year]
            unemployment = float(year_unrate.mean()) if not year_unrate.empty else None
        except Exception:
            unemployment = None
        result[year] = {"risk_free_rate": rf, "unemployment": unemployment}

    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _linear_score(value, good, bad, higher_is_better=True):
    """Map `value` onto 0-100 between the `bad` and `good` anchors."""
    if value is None:
        return None
    if good == bad:
        return 100.0 if (value >= good) == higher_is_better else 0.0
    frac = (value - bad) / (good - bad)
    return float(max(0.0, min(1.0, frac)) * 100)


def _average(scores):
    present = [s for s in scores if s is not None]
    return float(np.mean(present)) if present else None


def _rsi(closes: pd.Series, period: int = 14):
    if len(closes) < period + 1:
        return None
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _year_prices(daily_prices: pd.DataFrame, year: int) -> pd.DataFrame:
    return daily_prices[daily_prices.index.year == year]


def _price_on_or_before(daily_prices: pd.DataFrame, year: int, year_end: bool):
    """Year-end (Dec) or year-start (Jan) close, using the nearest available
    trading day -- a holiday/weekend boundary shouldn't drop the whole year."""
    year_rows = _year_prices(daily_prices, year)
    if year_rows.empty:
        return None
    return float(year_rows["Close"].iloc[-1]) if year_end else float(year_rows["Close"].iloc[0])


# ---------------------------------------------------------------------------
# Per-year PRISM score
# ---------------------------------------------------------------------------

def compute_prism_score(ticker_data: dict, year: int, macro: dict) -> dict:
    """Compute all 9 layers for a single fiscal year.

    ticker_data = {
        "ticker": str,
        "daily_prices": DataFrame (full OHLCV history),
        "fundamentals": dict from fetch_fundamentals_annual,
        "dividends": Series from fetch_dividends,
        "benchmark_prices": DataFrame OHLCV for the benchmark,
        "is_crypto": bool,
    }
    """
    daily_prices = ticker_data["daily_prices"]
    fundamentals = ticker_data.get("fundamentals") or {}
    dividends = ticker_data.get("dividends")
    benchmark_prices = ticker_data.get("benchmark_prices")
    is_crypto = ticker_data.get("is_crypto", False)

    fy = fundamentals.get(year, {}) if not is_crypto else {}
    prev_fy = fundamentals.get(year - 1, {}) if not is_crypto else {}

    price_history_to_date = daily_prices[daily_prices.index.year <= year]
    year_end_price = _price_on_or_before(daily_prices, year, year_end=True)
    year_start_price = _price_on_or_before(daily_prices, year, year_end=False)

    layers = {}

    if not is_crypto:
        # --- L1 Valuation ---
        shares = fy.get("shares_outstanding")
        net_income = fy.get("net_income")
        eps = (net_income / shares) if (net_income is not None and shares) else None
        pe = (year_end_price / eps) if (eps and eps > 0 and year_end_price) else None

        equity = fy.get("total_equity")
        bvps = (equity / shares) if (equity is not None and shares) else None
        pb = (year_end_price / bvps) if (bvps and bvps > 0 and year_end_price) else None

        market_cap = (year_end_price * shares) if (year_end_price and shares) else None
        ebitda = fy.get("ebitda")
        ev = (market_cap + (fy.get("total_debt") or 0) - (fy.get("cash") or 0)) if market_cap is not None else None
        ev_ebitda = (ev / ebitda) if (ev is not None and ebitda and ebitda > 0) else None

        layers["L1_valuation"] = _average([
            _linear_score(pe, good=15, bad=50, higher_is_better=False),
            _linear_score(pb, good=1.5, bad=5, higher_is_better=False),
            _linear_score(ev_ebitda, good=8, bad=25, higher_is_better=False),
        ])

        # --- L2 Profitability ---
        revenue = fy.get("revenue")
        margin = (net_income / revenue) if (net_income is not None and revenue) else None
        roe = (net_income / equity) if (net_income is not None and equity) else None
        assets = fy.get("total_assets")
        roa = (net_income / assets) if (net_income is not None and assets) else None

        layers["L2_profitability"] = _average([
            None if margin is None else max(0.0, min(1.0, margin / 0.20)) * 100,
            None if roe is None else max(0.0, min(1.0, roe / 0.25)) * 100,
            None if roa is None else max(0.0, min(1.0, roa / 0.12)) * 100,
        ])

        # --- L3 Growth ---
        prev_revenue = prev_fy.get("revenue")
        rev_growth = ((revenue - prev_revenue) / abs(prev_revenue)) if (revenue is not None and prev_revenue) else None

        prev_net_income = prev_fy.get("net_income")
        prev_shares = prev_fy.get("shares_outstanding")
        prev_eps = (prev_net_income / prev_shares) if (prev_net_income is not None and prev_shares) else None
        eps_growth = ((eps - prev_eps) / abs(prev_eps)) if (eps is not None and prev_eps) else None

        layers["L3_growth"] = _average([
            _linear_score(rev_growth, good=0.20, bad=-0.10, higher_is_better=True),
            _linear_score(eps_growth, good=0.20, bad=-0.10, higher_is_better=True),
        ])

        # --- L4 Safety ---
        debt = fy.get("total_debt")
        de_ratio = (debt / equity) if (debt is not None and equity) else None
        current_assets = fy.get("current_assets")
        current_liab = fy.get("current_liabilities")
        current_ratio = (current_assets / current_liab) if (current_assets is not None and current_liab) else None
        ebit = fy.get("ebit")
        interest_expense = fy.get("interest_expense")
        interest_coverage = (ebit / abs(interest_expense)) if (ebit is not None and interest_expense) else None

        layers["L4_safety"] = _average([
            _linear_score(de_ratio, good=0, bad=2.0, higher_is_better=False),
            _linear_score(current_ratio, good=2.5, bad=0.5, higher_is_better=True),
            _linear_score(interest_coverage, good=10, bad=1, higher_is_better=True),
        ])

        # --- L5 Efficiency ---
        asset_turnover = (revenue / assets) if (revenue is not None and assets) else None
        cogs = fy.get("cost_of_revenue")
        inventory = fy.get("inventory")
        inventory_turnover = (cogs / inventory) if (cogs is not None and inventory) else None
        op_cf = fy.get("operating_cashflow")
        ocf_ni = (op_cf / net_income) if (op_cf is not None and net_income and net_income > 0) else None

        layers["L5_efficiency"] = _average([
            None if asset_turnover is None else max(0.0, min(1.0, asset_turnover / 1.0)) * 100,
            None if inventory_turnover is None else max(0.0, min(1.0, inventory_turnover / 8.0)) * 100,
            _linear_score(ocf_ni, good=1.5, bad=0, higher_is_better=True),
        ])

        # --- L6 Dividends ---
        annual_dividend = None
        if dividends is not None and not dividends.empty:
            year_divs = dividends[dividends.index.year == year]
            annual_dividend = float(year_divs.sum()) if not year_divs.empty else 0.0
        div_yield = (annual_dividend / year_end_price) if (annual_dividend and year_end_price) else None
        payout_ratio = (annual_dividend / eps) if (annual_dividend and eps and eps > 0) else None

        if not annual_dividend:
            layers["L6_dividends"] = 40.0
        else:
            yield_score = max(0.0, min(1.0, (div_yield or 0) / 0.04)) * 100
            if payout_ratio is None:
                payout_score = None
            elif 0.20 <= payout_ratio <= 0.50:
                payout_score = 100.0
            elif payout_ratio > 0.80:
                payout_score = 20.0
            else:
                payout_score = 70.0
            layers["L6_dividends"] = _average([yield_score, payout_score])
    else:
        layers.update({k: None for k in ["L1_valuation", "L2_profitability", "L3_growth", "L4_safety", "L5_efficiency", "L6_dividends"]})

    # --- L7 Momentum (price-based, works for crypto too) ---
    price_return = ((year_end_price / year_start_price) - 1) if (year_end_price and year_start_price) else None
    rsi_val = _rsi(price_history_to_date["Close"]) if not price_history_to_date.empty else None
    rsi_score = None
    if rsi_val is not None:
        if rsi_val > 70 or rsi_val < 30:
            rsi_score = max(0.0, 100 - abs(rsi_val - 55) * 2)
        else:
            rsi_score = 100 - abs(rsi_val - 55) * 1.2

    golden_cross_bonus = 0.0
    if len(price_history_to_date) >= 200:
        ma50 = price_history_to_date["Close"].rolling(50).mean().iloc[-1]
        ma200 = price_history_to_date["Close"].rolling(200).mean().iloc[-1]
        if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
            golden_cross_bonus = 10.0

    momentum_score = _average([
        _linear_score(price_return, good=0.30, bad=-0.20, higher_is_better=True),
        rsi_score,
    ])
    layers["L7_momentum"] = None if momentum_score is None else min(100.0, momentum_score + golden_cross_bonus)

    # --- L8 ESG ---
    esg_score = ticker_data.get("esg_score")
    layers["L8_esg"] = 50.0 if esg_score is None else esg_score

    # --- L9 Relative Strength ---
    rel_score = None
    if benchmark_prices is not None and not benchmark_prices.empty and year_start_price and year_end_price:
        bench_year = _year_prices(benchmark_prices, year)
        if not bench_year.empty:
            bench_start = float(bench_year["Close"].iloc[0])
            bench_end = float(bench_year["Close"].iloc[-1])
            if bench_start:
                ticker_ret = (year_end_price / year_start_price) - 1
                bench_ret = (bench_end / bench_start) - 1
                outperformance = (ticker_ret - bench_ret) * 100  # percentage points
                rel_score = _linear_score(outperformance, good=20, bad=-20, higher_is_better=True)
    layers["L9_relative_strength"] = rel_score

    # --- Weighted composite, renormalized over whatever is available ---
    weights = NO_FUNDAMENTALS_WEIGHTS if is_crypto else LAYER_WEIGHTS
    available = {k: v for k, v in layers.items() if v is not None and k in weights}
    total_weight = sum(weights[k] for k in available)
    score = sum(weights[k] * v for k, v in available.items()) / total_weight if total_weight else 50.0

    n_layers = len([v for v in layers.values() if v is not None])
    if n_layers >= 8:
        data_quality = "full"
    elif n_layers >= 4:
        data_quality = "partial"
    else:
        data_quality = "minimal"

    return {
        "score": round(score, 1),
        "signal": None,  # filled in by the caller once thresholds are known
        "layers": layers,
        "data_quality": data_quality,
        "years_available": n_layers,
    }


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
    is_crypto = ticker.upper().endswith("-USD")

    daily_prices = fetch_stock_data(ticker, period_years)
    benchmark_prices = fetch_stock_data(benchmark, period_years)

    # A target ticker's own history can predate the benchmark's existence
    # (e.g. MSFT back to 1986 vs QQQ's 1999 inception, or a 10y lookback on
    # a ticker with only 3y of real history). Reindexing the benchmark onto
    # dates before it existed leaves leading NaNs with nothing to
    # forward-fill from, which poisons every benchmark stat (total return,
    # CAGR, Sharpe all come back NaN). Clip to the actual overlap instead --
    # this also means the benchmark's own stats are computed over the same
    # window as whatever ticker it's being compared against, not blindly
    # reindexed onto a window it didn't trade through.
    common_start = max(daily_prices.index[0], benchmark_prices.index[0])
    common_end = min(daily_prices.index[-1], benchmark_prices.index[-1])
    if common_start >= common_end:
        raise ValueError(
            f"'{ticker}' and benchmark '{benchmark}' have no overlapping price history."
        )
    daily_prices = daily_prices[(daily_prices.index >= common_start) & (daily_prices.index <= common_end)]

    fundamentals = {} if is_crypto else fetch_fundamentals_annual(ticker)
    dividends = pd.Series(dtype=float) if is_crypto else fetch_dividends(ticker)
    esg_score = None if is_crypto else fetch_sustainability_score(ticker)

    start_year = int(daily_prices.index[0].year)
    end_year = int(daily_prices.index[-1].year)
    macro = fetch_macro_data(start_year, end_year)

    ticker_data = {
        "ticker": ticker,
        "daily_prices": daily_prices,
        "fundamentals": fundamentals,
        "dividends": dividends,
        "benchmark_prices": benchmark_prices,
        "esg_score": esg_score,
        "is_crypto": is_crypto,
    }

    rows = []
    for year in range(start_year, end_year + 1):
        result = compute_prism_score(ticker_data, year, macro.get(year, {}))
        result["signal"] = score_to_signal(result["score"], buy_threshold, sell_threshold)
        row = {"year": year, "score": result["score"], "signal": result["signal"],
               "data_quality": result["data_quality"], "years_available": result["years_available"]}
        row.update(result["layers"])
        rows.append(row)
    scores_df = pd.DataFrame(rows).set_index("year")

    # --- Simulate the strategy at each rebalance date ---
    rebal_dates = _rebalance_dates(daily_prices, rebalance)
    trade_log = []
    invested = False
    entry_price = None
    entry_date = None

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
            entry_date = date
            trade_log.append({"date": date, "action": "BUY", "price": price_at_date, "score": score})
        elif signal == "SELL" and invested:
            pnl_pct = (price_at_date / entry_price - 1) * 100
            trade_log.append({"date": date, "action": "SELL", "price": price_at_date, "score": score, "pnl_pct": pnl_pct})
            invested = False
            entry_price = None
            entry_date = None

    # --- Equity curves ---
    idx = daily_prices.index
    equity = pd.Series(index=idx, dtype=float)
    equity_val = 100.0
    is_invested = False
    entry_px = None
    entry_equity = None
    trade_iter = iter(trade_log)
    next_trade = next(trade_iter, None)
    risk_free_daily = (macro.get(start_year, {}).get("risk_free_rate", DEFAULT_RISK_FREE_RATE)) / 252

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
        "scores": scores_df,
        "daily_prices": daily_prices,
        "benchmark_prices": benchmark_prices,
        "trade_log": trade_log,
        "equity_curve": equity,
        "benchmark_curve": benchmark_curve,
        "stats": stats,
        "ticker": ticker,
        "benchmark": benchmark,
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
