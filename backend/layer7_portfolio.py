"""
Layer 7 — Portfolio Fit
User inputs their portfolio (ticker + shares held).
S7 measures how well the target stock fits into THAT portfolio:
  - Correlation with portfolio (low = good diversification)
  - Concentration risk (how much of portfolio this stock would be)
  - Portfolio beta contribution
"""

import functools
import numpy as np
import pandas as pd
import yfinance as yf

@functools.lru_cache(maxsize=32)
def _returns(ticker: str) -> pd.Series:
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    # .squeeze() alone collapses to a bare scalar (not a Series) if yfinance
    # ever returns a single row for this ticker -- only squeeze the column
    # axis, never the row axis.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.pct_change().dropna()

def _current_price(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df.empty:
            return 0.0
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return float(close.iloc[-1])

def compute_layer7(target_ticker: str, portfolio: list[dict]) -> dict:
    """
    portfolio: list of {"ticker": str, "shares": float}
    Returns S7 and details.
    """
    if not portfolio:
        return {"S7": 50, "data_quality": "stub",
                "details": ["No portfolio provided — score defaulted to neutral (50)"]}

    # Build portfolio value weights
    holdings = {}
    total_value = 0.0
    for item in portfolio:
        t = item["ticker"].upper()
        s = float(item["shares"])
        price = _current_price(t)
        val = price * s
        holdings[t] = {"shares": s, "price": price, "value": val}
        total_value += val

    if total_value == 0:
        return {"S7": 50, "data_quality": "stub", "details": ["Could not fetch portfolio prices"]}

    for t in holdings:
        holdings[t]["weight"] = holdings[t]["value"] / total_value

    # Target price (for concentration calc — assume buying 1 lot / $10k notional)
    target_price = _current_price(target_ticker)

    # Fetch returns
    target_returns = _returns(target_ticker)
    port_returns_series = []
    for t, h in holdings.items():
        r = _returns(t)
        if not r.empty:
            port_returns_series.append(r * h["weight"])

    if not port_returns_series or target_returns.empty:
        return {"S7": 50, "data_quality": "partial",
                "details": ["Insufficient return data for correlation calculation"]}

    port_returns = pd.concat(port_returns_series, axis=1).sum(axis=1)
    idx = target_returns.index.intersection(port_returns.index)
    if len(idx) < 40:
        return {"S7": 50, "data_quality": "partial", "details": ["Too few overlapping trading days"]}

    tr = target_returns.loc[idx]
    pr = port_returns.loc[idx]

    # 1. Correlation score (low correlation = good fit) — weight 0.45
    corr = float(tr.corr(pr))
    if   corr < 0.2:   corr_score = 90  # great diversifier
    elif corr < 0.4:   corr_score = 75
    elif corr < 0.6:   corr_score = 55
    elif corr < 0.8:   corr_score = 35
    else:              corr_score = 15  # highly correlated, little diversification

    # 2. Concentration risk (if adding $10k worth) — weight 0.35
    add_value = min(10000, total_value * 0.10)  # assume adding ~10% or $10k
    new_weight = add_value / (total_value + add_value)
    existing_max = max(h["weight"] for h in holdings.values())
    new_max = max(existing_max, new_weight)
    if   new_max < 0.10: conc_score = 90
    elif new_max < 0.20: conc_score = 70
    elif new_max < 0.35: conc_score = 50
    elif new_max < 0.50: conc_score = 30
    else:                conc_score = 10

    # 3. Beta vs portfolio — weight 0.20
    cov_matrix = np.cov(tr.values, pr.values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
    if   0.5 <= beta <= 1.2: beta_score = 80  # reasonable range
    elif 0.2 <= beta < 0.5:  beta_score = 70  # low beta, defensive
    elif 1.2 < beta <= 1.8:  beta_score = 55  # high beta, aggressive
    elif beta < 0:           beta_score = 60  # negative beta = hedge
    else:                    beta_score = 30  # very high beta

    S7 = corr_score * 0.45 + conc_score * 0.35 + beta_score * 0.20

    top_holdings = sorted(holdings.items(), key=lambda x: x[1]["weight"], reverse=True)[:3]
    top_str = ", ".join(f"{t} ({h['weight']*100:.1f}%)" for t, h in top_holdings)

    details = [
        f"Portfolio correlation with {target_ticker}: {corr:.2f} ({'low — good diversifier' if corr < 0.4 else 'moderate' if corr < 0.7 else 'high — low diversification'})",
        f"Estimated position weight if added: {new_weight*100:.1f}%",
        f"Beta of {target_ticker} vs portfolio: {beta:.2f}",
        f"Top holdings: {top_str}",
        f"Portfolio total value: ${total_value:,.0f}",
    ]

    return {
        "S7": round(S7, 1),
        "correlation": round(corr, 3),
        "beta": round(beta, 3),
        "new_position_weight": round(new_weight, 3),
        "portfolio_value": round(total_value, 2),
        "holdings": holdings,
        "details": details,
        "data_quality": "full",
    }
