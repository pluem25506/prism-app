"""
Layer 2 (S2): Value Creation.

Scores a stock on whether it actually creates economic value -- does its
return on invested capital (ROIC) clear its cost of capital (WACC), convert
that into real free cash flow relative to its market price, and do so
efficiently (ROCE, Gross Margin, Receivables Days) and growingly (Revenue
CAGR).

Works for both the live Analyze page (year=None) and the Backtest engine
(an explicit historical year) via the shared fetch_fundamentals_annual()
dict; live mode additionally uses today's actual price/beta for WACC, while
historical mode reconstructs both from price history up to that year-end.
"""

from .data_fetcher import (
    fetch_fundamentals_annual, fetch_price_history_long, get_benchmark_ticker,
    slice_price_window, _linear_score, weighted_average,
)

_RF = 0.028  # Thai 10Y government bond yield (live-mode fallback)
_ERP = 0.065  # Damodaran Thailand equity risk premium (emerging market)
_DEFAULT_TAX_RATE = 0.20
_ESTIMATED_WACC = 0.09


def _year_fy(fundamentals, year):
    if year is None:
        if not fundamentals:
            return {}, {}, None
        year = max(fundamentals.keys())
    return fundamentals.get(year, {}), fundamentals.get(year - 1, {}), year


def _compute_tax_rate(fy):
    tax_expense = fy.get("tax_expense")
    pretax_income = fy.get("pretax_income")
    if tax_expense is None or not pretax_income:
        return _DEFAULT_TAX_RATE, False
    rate = tax_expense / pretax_income
    if rate < 0 or rate > 0.6:
        return _DEFAULT_TAX_RATE, False
    return rate, True


def _compute_roic(fy, tax_rate):
    ebit = fy.get("ebit")
    total_equity = fy.get("total_equity")
    cash = fy.get("cash")
    if ebit is None or total_equity is None or cash is None:
        return None, {}
    total_debt = (fy.get("short_term_debt") or 0) + (fy.get("long_term_debt") or 0)
    nopat = ebit * (1 - tax_rate)
    invested_capital = total_equity + total_debt - cash
    if not invested_capital:
        return None, {}
    roic = nopat / invested_capital
    return roic, {"ebit": ebit, "nopat": nopat, "invested_capital": invested_capital}


def _historical_beta(ticker, price_history, year):
    """Beta vs the stock's own benchmark, computed from daily returns up to
    `year`'s year-end -- the historical analogue of info['beta'] (which has
    no historical dimension, only a single live value)."""
    try:
        window = slice_price_window(price_history, year)
        bench = fetch_price_history_long(get_benchmark_ticker(ticker))
        bench_window = slice_price_window(bench, year)
        if window.empty or bench_window.empty:
            return 1.0
        stock_ret = window["Close"].pct_change().dropna()
        bench_ret = bench_window["Close"].pct_change().dropna()
        aligned = stock_ret.align(bench_ret, join="inner")
        if len(aligned[0]) < 20:
            return 1.0
        cov = aligned[0].cov(aligned[1])
        var = aligned[1].var()
        return float(cov / var) if var else 1.0
    except Exception:
        return 1.0


def _compute_wacc(fy, tax_rate, beta, price, shares):
    ke = _RF + beta * _ERP
    interest_expense = fy.get("interest_expense")
    total_debt = (fy.get("short_term_debt") or 0) + (fy.get("long_term_debt") or 0)
    market_cap = (price * shares) if (price and shares) else None

    if interest_expense is None or total_debt <= 0 or not market_cap:
        return None, {"beta": beta, "ke": ke}, False

    kd = abs(interest_expense) / total_debt
    kd_after_tax = kd * (1 - tax_rate)
    v = market_cap + total_debt
    we = market_cap / v
    wd = total_debt / v
    wacc = we * ke + wd * kd_after_tax
    return wacc, {"beta": beta, "ke": ke, "kd": kd, "market_cap": market_cap}, True


def _map_spread(spread):
    if spread is None:
        return None
    if spread >= 0.10:
        return 100.0
    if spread >= 0.05:
        return 80.0
    if spread >= 0.00:
        return 60.0
    if spread >= -0.05:
        return 30.0
    return 0.0


def _map_fcf_yield(fcf_yield):
    if fcf_yield is None:
        return None
    if fcf_yield >= 0.08:
        return 100.0
    if fcf_yield >= 0.04:
        return 70.0
    if fcf_yield >= 0.00:
        return 40.0
    return 0.0


def calculate_value_score(ticker: str, year: int | None = None) -> dict:
    """Compute S2 (value creation, 0-100): ROIC-WACC spread (30%), FCF
    Yield (20%), ROCE (20%), Revenue CAGR (15%), Gross Margin (10%),
    Receivables Days (5%).

    year=None (live) uses today's actual price/shares for WACC's market cap
    and the stock's live beta; an explicit year (Backtest) reconstructs both
    from price history ending at that fiscal year, so the two modes run the
    identical formula pointed at different points in time.
    """
    fundamentals = fetch_fundamentals_annual(ticker)
    fy, prev_fy, resolved_year = _year_fy(fundamentals, year)
    price_history = fetch_price_history_long(ticker)

    if year is None:
        # Live: use today's actual price (last close) and a live-style beta
        # (computed the same historical way, just "as of now" i.e. no cutoff).
        price = float(price_history["Close"].iloc[-1]) if not price_history.empty else None
        beta = _historical_beta(ticker, price_history, None)
    else:
        year_rows = price_history[price_history.index.year == year]
        price = float(year_rows["Close"].iloc[-1]) if not year_rows.empty else None
        beta = _historical_beta(ticker, price_history, year)

    shares = fy.get("shares_outstanding")

    tax_rate, tax_rate_computed = _compute_tax_rate(fy)
    roic, roic_raw = _compute_roic(fy, tax_rate)
    wacc, wacc_raw, wacc_computed = _compute_wacc(fy, tax_rate, beta, price, shares)
    estimated_wacc = not wacc_computed
    if wacc is None:
        wacc = _ESTIMATED_WACC

    spread = (roic - wacc) if roic is not None else None

    ocf = fy.get("operating_cashflow")
    capex = fy.get("capex")
    market_cap = (price * shares) if (price and shares) else None
    # yfinance reports Capital Expenditure as a negative cash OUTFLOW, so
    # FCF = OCF + capex (adding a negative correctly subtracts it) -- not
    # OCF - capex, which would double-add it back. Confirmed by direct
    # inspection: every ticker checked came back with capex < 0.
    fcf = (ocf + capex) if (ocf is not None and capex is not None) else None
    fcf_yield = (fcf / market_cap) if (fcf is not None and market_cap) else None

    revenue = fy.get("revenue")
    assets = fy.get("total_assets")
    current_liab = fy.get("current_liabilities")
    ebit = fy.get("ebit")
    capital_employed = (assets - current_liab) if (assets is not None and current_liab is not None) else None
    roce = (ebit / capital_employed) if (ebit is not None and capital_employed and capital_employed > 0) else None

    cogs = fy.get("cost_of_revenue")
    gross_margin = ((revenue - cogs) / revenue) if (revenue and cogs is not None and revenue > 0) else None

    cagr = None
    if revenue and resolved_year is not None:
        for n_back in range(5, 0, -1):
            base_revenue = fundamentals.get(resolved_year - n_back, {}).get("revenue")
            if base_revenue and base_revenue > 0:
                cagr = (revenue / base_revenue) ** (1 / n_back) - 1
                break

    receivables = fy.get("accounts_receivable")
    receivables_days = (receivables / revenue * 365) if (receivables is not None and revenue and revenue > 0) else None

    spread_score = _map_spread(spread)
    fcf_yield_score = _map_fcf_yield(fcf_yield)
    roce_score = _linear_score(roce, good=0.20, bad=0.0, higher_is_better=True)
    cagr_score = _linear_score(cagr, good=0.15, bad=-0.05, higher_is_better=True)
    gross_margin_score = None if gross_margin is None else max(0.0, min(1.0, gross_margin / 0.40)) * 100
    receivables_score = _linear_score(receivables_days, good=30, bad=90, higher_is_better=False)

    s2 = weighted_average([
        (spread_score, 0.30), (fcf_yield_score, 0.20), (roce_score, 0.20),
        (cagr_score, 0.15), (gross_margin_score, 0.10), (receivables_score, 0.05),
    ])

    details = {
        "roic": roic_raw, "wacc": wacc_raw, "tax_rate": tax_rate, "tax_rate_computed": tax_rate_computed,
        "spread_score": spread_score, "fcf_yield_score": fcf_yield_score, "roce_score": roce_score,
        "cagr_score": cagr_score, "gross_margin_score": gross_margin_score, "receivables_score": receivables_score,
        "year": resolved_year,
    }

    return {
        "S2": round(s2, 2) if s2 is not None else None,
        "roic": round(roic, 4) if roic is not None else None,
        "wacc": round(wacc, 4) if wacc is not None else None,
        "spread": round(spread, 4) if spread is not None else None,
        "fcf_yield": round(fcf_yield, 4) if fcf_yield is not None else None,
        "roce": round(roce, 4) if roce is not None else None,
        "gross_margin": round(gross_margin, 4) if gross_margin is not None else None,
        "revenue_cagr": round(cagr, 4) if cagr is not None else None,
        "receivables_days": round(receivables_days, 1) if receivables_days is not None else None,
        "estimated_wacc": estimated_wacc,
        "details": details,
    }
