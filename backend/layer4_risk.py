"""
Layer 4 (S4): Risk.

Computes historical VaR/CVaR, beta vs a market benchmark, the Sharpe ratio,
and GARCH(1,1) conditional volatility from a ticker's daily returns, then
combines Sharpe/VaR/beta into a single 0-100 risk score.

Works for both the live Analyze page (year=None -> trailing 1y from today)
and the Backtest engine (an explicit year -> trailing 1y ending at that
fiscal year-end), from the same long price history sliced to the right
window, instead of always re-fetching "the last year" fresh.

Fixed along the way: beta used to compare against ^SET.BK, which yfinance
only ever returns a single day of history for (confirmed by direct testing,
not a transient outage) -- so Thai-ticker beta always silently fell back to
info["beta"] (or 1.0). Now uses THD (iShares MSCI Thailand ETF), the same
fix already applied to L9's RS Ratio.
"""

import numpy as np
import pandas as pd
from arch import arch_model

from .data_fetcher import fetch_price_history_long, get_benchmark_ticker, slice_price_window

_RF_ANNUAL = 0.028
_TRADING_DAYS = 252


def _returns_for(ticker, price_history, as_of):
    window = slice_price_window(price_history, as_of)
    if window.empty:
        return pd.Series(dtype=float)
    return window["Close"].pct_change().dropna()


def _compute_var_cvar(returns):
    if returns is None or len(returns) == 0:
        return None, None, None, None
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    tail_95 = returns[returns <= var_95]
    tail_99 = returns[returns <= var_99]
    cvar_95 = float(tail_95.mean()) if len(tail_95) > 0 else var_95
    cvar_99 = float(tail_99.mean()) if len(tail_99) > 0 else var_99
    return var_95, var_99, cvar_95, cvar_99


def _compute_beta(ticker, returns, as_of):
    """Beta vs THD (Thai tickers) or ^GSPC (everything else), over the same
    window as `returns`."""
    try:
        benchmark = get_benchmark_ticker(ticker)
        bench_history = fetch_price_history_long(benchmark)
        bench_window = slice_price_window(bench_history, as_of)
        if bench_window.empty:
            raise ValueError("no benchmark data")
        bench_returns = bench_window["Close"].pct_change().dropna()

        aligned = pd.concat([returns, bench_returns], axis=1).dropna()
        if len(aligned) < 20:
            raise ValueError("insufficient overlapping benchmark data")

        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var = np.var(aligned.iloc[:, 1])
        if var == 0:
            raise ZeroDivisionError("zero benchmark variance")
        return float(cov / var), f"vs_{benchmark}"
    except Exception:
        return 1.0, "fallback_1.0"


def _compute_sharpe(returns):
    if returns is None or len(returns) == 0:
        return None
    rf_daily = _RF_ANNUAL / _TRADING_DAYS
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None
    return float((returns.mean() - rf_daily) / std * np.sqrt(_TRADING_DAYS))


def _compute_garch_vol(returns):
    if returns is None or len(returns) < 30:
        return None, "insufficient_data"
    try:
        am = arch_model(returns * 100, vol="Garch", p=1, q=1)
        res = am.fit(disp="off")
        variance = res.forecast(horizon=1).variance.iloc[-1, 0]
        garch_vol = (variance ** 0.5) / 100 * np.sqrt(_TRADING_DAYS)
        return float(garch_vol), "garch"
    except Exception:
        return float(returns.std() * np.sqrt(_TRADING_DAYS)), "historical_fallback"


def _map_sharpe(sharpe):
    if sharpe is None:
        return 20.0
    if sharpe >= 1.5:
        return 100.0
    if sharpe >= 0.5:
        return 60.0
    return 20.0


def _map_var(var_95):
    if var_95 is None:
        return 20.0
    if var_95 >= -0.02:
        return 100.0
    if var_95 >= -0.04:
        return 60.0
    return 20.0


def _map_beta(beta):
    if beta is None:
        return 20.0
    if 0.5 <= beta <= 1.2:
        return 100.0
    if beta < 1.5:
        return 60.0
    return 20.0


def calculate_risk_score(ticker: str, year: int | None = None, as_of_date=None) -> dict:
    """Compute S4 (risk, 0-100) from Sharpe (40%), historical VaR (30%), and
    beta vs the market (30%), over the trailing year ending today (live),
    ending at a historical fiscal year (Backtest, annual granularity), or
    ending at a specific rebalance date (Backtest, sub-annual granularity).

    as_of_date, when given, takes priority over `year` for the price
    window's end point -- `year` still selects which fiscal year's
    fundamentals-based layers (L1/L2/L3/L5/L8) apply, but L4 is purely
    price-based and can refresh more often than once a year.
    """
    price_history = fetch_price_history_long(ticker)
    as_of = as_of_date if as_of_date is not None else year
    returns = _returns_for(ticker, price_history, as_of)

    var_95, var_99, cvar_95, cvar_99 = _compute_var_cvar(returns)
    beta, beta_source = _compute_beta(ticker, returns, as_of)
    sharpe = _compute_sharpe(returns)
    garch_vol, garch_source = _compute_garch_vol(returns)

    sharpe_score = _map_sharpe(sharpe)
    var_score = _map_var(var_95)
    beta_score = _map_beta(beta)
    s4 = sharpe_score * 0.40 + var_score * 0.30 + beta_score * 0.30

    details = {
        "sharpe_score": sharpe_score, "var_score": var_score, "beta_score": beta_score,
        "beta_source": beta_source, "garch_source": garch_source,
        "n_observations": int(len(returns)) if returns is not None else 0,
        "year": year,
    }

    return {
        "S4": round(s4, 2),
        "VaR_95": var_95, "VaR_99": var_99, "CVaR_95": cvar_95, "CVaR_99": cvar_99,
        "beta": beta, "sharpe": sharpe, "garch_vol_annualized": garch_vol,
        "details": details,
    }
