"""
Layer 5 (S5): Monte Carlo price simulation.

Simulates 1-year-ahead price paths using a Student-t return distribution
(df=5, fat tails) and scores the outcome on expected upside, penalized by
the width of the outcome distribution (uncertainty).
"""

import numpy as np
from scipy.stats import t as student_t

from .data_fetcher import get_price_series, get_return_series

_N_SIMULATIONS = 10000
_N_DAYS = 252
_DEGREES_OF_FREEDOM = 5  # fixed per PRISM v2.1 spec
_PATHS_SAMPLE_SIZE = 100


def calculate_montecarlo_score(ticker: str) -> dict:
    """Compute S5 (Monte Carlo outlook, 0-100) for a ticker from a 10,000-path,
    252-day Student-t simulation of daily returns."""
    prices = get_price_series(ticker)
    returns = get_return_series(ticker)

    if prices.empty or returns.empty:
        return {
            "S5": None,
            "current_price": None,
            "P10": None,
            "P25": None,
            "P50": None,
            "P75": None,
            "P90": None,
            "prob_loss": None,
            "paths_sample": None,
            "upside_P50": None,
            "details": {"error": "insufficient price history"},
        }

    current_price = float(prices.iloc[-1])
    mu = float(returns.mean())
    sigma = float(returns.std())

    np.random.seed(42)
    simulated_returns = student_t.rvs(
        df=_DEGREES_OF_FREEDOM, loc=mu, scale=sigma, size=(_N_SIMULATIONS, _N_DAYS)
    )
    price_paths = current_price * np.cumprod(1 + simulated_returns, axis=1)
    final_prices = price_paths[:, -1]

    p10 = float(np.percentile(final_prices, 10))
    p25 = float(np.percentile(final_prices, 25))
    p50 = float(np.percentile(final_prices, 50))
    p75 = float(np.percentile(final_prices, 75))
    p90 = float(np.percentile(final_prices, 90))
    prob_loss = float((final_prices < current_price).mean())

    paths_sample = price_paths[np.random.choice(_N_SIMULATIONS, _PATHS_SAMPLE_SIZE, replace=False)]

    upside = (p50 - current_price) / current_price
    width = (p90 - p10) / current_price

    if upside >= 0.25:
        base_score = 90.0
    elif upside >= 0.15:
        base_score = 75.0
    elif upside >= 0.05:
        base_score = 60.0
    elif upside >= 0.0:
        base_score = 45.0
    elif upside >= -0.10:
        base_score = 30.0
    else:
        base_score = 15.0

    loss_penalty = prob_loss * 20.0
    width_penalty = min(15.0, (width / 3.0) * 15.0)
    s5 = max(0.0, base_score - loss_penalty - width_penalty)

    details = {
        "mu_daily": mu,
        "sigma_daily": sigma,
        "n_simulations": _N_SIMULATIONS,
        "n_days": _N_DAYS,
        "degrees_of_freedom": _DEGREES_OF_FREEDOM,
        "base_score": base_score,
        "loss_penalty": loss_penalty,
        "width_penalty": width_penalty,
        "width": width,
    }

    return {
        "S5": round(s5, 2),
        "current_price": current_price,
        "P10": p10,
        "P25": p25,
        "P50": p50,
        "P75": p75,
        "P90": p90,
        "prob_loss": prob_loss,
        "paths_sample": paths_sample,
        "upside_P50": upside,
        "details": details,
    }
