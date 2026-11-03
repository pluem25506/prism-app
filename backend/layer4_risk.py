"""
Layer 4 (S4): Risk.

Computes historical VaR/CVaR, beta vs the SET index, the Sharpe ratio (Thai
risk-free rate), and GARCH(1,1) conditional volatility from a ticker's daily
returns, then combines Sharpe/VaR/beta into a single 0-100 risk score.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

from .data_fetcher import fetch_stock_data, get_price_series, get_return_series

_RF_ANNUAL = 0.028
_TRADING_DAYS = 252


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


def _compute_beta(ticker, returns):
    """Beta vs the SET index, falling back to stock.info's beta if the
    index download fails or there isn't enough overlapping history."""
    try:
        set_data = yf.download("^SET.BK", period="1y", progress=False)
        set_close = set_data["Close"]
        if isinstance(set_close, pd.DataFrame):
            set_close = set_close.iloc[:, 0]
        set_returns = set_close.pct_change().dropna()

        aligned = pd.concat([returns, set_returns], axis=1).dropna()
        if len(aligned) < 2:
            raise ValueError("insufficient overlapping SET index data")

        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        var = np.var(aligned.iloc[:, 1])
        if var == 0:
            raise ZeroDivisionError("zero SET index variance")

        return float(cov / var), "set_index"
    except Exception:
        info = fetch_stock_data(ticker).get("info") or {}
        beta = info.get("beta", 1.0)
        if beta is None:
            beta = 1.0
        return float(beta), "info_fallback"


def _compute_sharpe(returns):
    if returns is None or len(returns) == 0:
        return None

    rf_daily = _RF_ANNUAL / _TRADING_DAYS
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None

    sharpe = (returns.mean() - rf_daily) / std * np.sqrt(_TRADING_DAYS)
    return float(sharpe)


def _compute_garch_vol(returns):
    """Annualized GARCH(1,1) conditional volatility, with a historical-vol
    fallback if the model fails to fit (e.g. too little data)."""
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


def calculate_risk_score(ticker: str) -> dict:
    """Compute S4 (risk, 0-100) for a ticker from Sharpe, historical VaR, and
    beta vs the SET index, weighted 40/30/30."""
    prices = get_price_series(ticker)
    returns = get_return_series(ticker)

    var_95, var_99, cvar_95, cvar_99 = _compute_var_cvar(returns)
    beta, beta_source = _compute_beta(ticker, returns)
    sharpe = _compute_sharpe(returns)
    garch_vol, garch_source = _compute_garch_vol(returns)

    sharpe_score = _map_sharpe(sharpe)
    var_score = _map_var(var_95)
    beta_score = _map_beta(beta)

    s4 = sharpe_score * 0.40 + var_score * 0.30 + beta_score * 0.30

    details = {
        "sharpe_score": sharpe_score,
        "var_score": var_score,
        "beta_score": beta_score,
        "beta_source": beta_source,
        "garch_source": garch_source,
        "n_observations": int(len(returns)) if returns is not None else 0,
    }

    return {
        "S4": round(s4, 2),
        "VaR_95": var_95,
        "VaR_99": var_99,
        "CVaR_95": cvar_95,
        "CVaR_99": cvar_99,
        "beta": beta,
        "sharpe": sharpe,
        "garch_vol_annualized": garch_vol,
        "details": details,
    }
