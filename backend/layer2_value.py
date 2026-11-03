"""
Layer 2 (S2): Value Creation.

Scores a stock on whether it actually creates economic value: does its
return on invested capital (ROIC) clear its cost of capital (WACC), and does
it convert that into real free cash flow relative to its market price.
"""

from .data_fetcher import fetch_stock_data, get_financial_item

_RF = 0.028  # Thai 10Y government bond yield
_ERP = 0.065  # Damodaran Thailand equity risk premium (emerging market)
_DEFAULT_TAX_RATE = 0.20
_ESTIMATED_WACC = 0.09  # industry median fallback

_KW_EBIT = ["EBIT"]
_KW_OPERATING_INCOME = ["Operating Income"]
_KW_INCOME_TAX_EXPENSE = ["Tax Provision", "Income Tax Expense"]
_KW_PRETAX_INCOME = ["Pretax Income", "Income Before Tax"]
_KW_INTEREST_EXPENSE = ["Interest Expense"]
_KW_TOTAL_EQUITY = [
    "Total Equity Gross Minority Interest",
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Common Stock Equity",
]
_KW_SHORT_TERM_DEBT = ["Current Debt", "Short Term Debt"]
# NOTE: "Short Long Term Debt" is deliberately excluded -- its normalized form
# contains "longtermdebt" as a substring, which causes data_fetcher's fuzzy
# matcher to collide with the "Long Term Debt" row and double-count debt.
_KW_LONG_TERM_DEBT = ["Long Term Debt", "LongTermDebt"]
_KW_CASH = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]
_KW_OPERATING_CASH_FLOW = [
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]
_KW_CAPEX = [
    "Capital Expenditure",
    "Capital Expenditures",
    "Purchase Of PPE",
    "Purchases Of Property Plant And Equipment",
]


def _first_available(df, *keyword_groups):
    for keywords in keyword_groups:
        value = get_financial_item(df, keywords)
        if value is not None:
            return value
    return None


def _compute_tax_rate(fin):
    tax_expense = get_financial_item(fin, _KW_INCOME_TAX_EXPENSE)
    pretax_income = get_financial_item(fin, _KW_PRETAX_INCOME)
    if tax_expense is None or pretax_income is None or pretax_income == 0:
        return _DEFAULT_TAX_RATE, False

    rate = tax_expense / pretax_income
    if rate < 0 or rate > 0.6:
        return _DEFAULT_TAX_RATE, False
    return rate, True


def _compute_roic(bs, fin, tax_rate):
    ebit = _first_available(fin, _KW_EBIT, _KW_OPERATING_INCOME)
    total_equity = get_financial_item(bs, _KW_TOTAL_EQUITY)
    cash = get_financial_item(bs, _KW_CASH)

    if ebit is None or total_equity is None or cash is None:
        return None, {}

    short_term_debt = get_financial_item(bs, _KW_SHORT_TERM_DEBT) or 0
    long_term_debt = get_financial_item(bs, _KW_LONG_TERM_DEBT) or 0
    total_debt = short_term_debt + long_term_debt

    nopat = ebit * (1 - tax_rate)
    invested_capital = total_equity + total_debt - cash
    if invested_capital == 0:
        return None, {}

    roic = nopat / invested_capital
    raw = {
        "ebit": ebit,
        "nopat": nopat,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "cash": cash,
        "invested_capital": invested_capital,
    }
    return roic, raw


def _compute_wacc(bs, fin, info, tax_rate):
    beta = info.get("beta")
    if beta is None:
        beta = 1.0
    ke = _RF + beta * _ERP

    interest_expense = get_financial_item(fin, _KW_INTEREST_EXPENSE)
    short_term_debt = get_financial_item(bs, _KW_SHORT_TERM_DEBT) or 0
    long_term_debt = get_financial_item(bs, _KW_LONG_TERM_DEBT) or 0
    total_debt = short_term_debt + long_term_debt
    market_cap = info.get("marketCap")

    if interest_expense is None or total_debt <= 0 or market_cap is None or market_cap <= 0:
        # Not enough debt/interest data for a real WACC -- caller falls back
        # to the industry median.
        return None, {"beta": beta, "ke": ke}, False

    # abs() because yfinance reports Interest Expense as a negative outflow;
    # Kd is a cost (rate), so it must come out positive.
    kd = abs(interest_expense) / total_debt
    kd_after_tax = kd * (1 - tax_rate)

    v = market_cap + total_debt
    we = market_cap / v
    wd = total_debt / v
    wacc = we * ke + wd * kd_after_tax

    raw = {
        "beta": beta,
        "ke": ke,
        "interest_expense": interest_expense,
        "total_debt": total_debt,
        "kd": kd,
        "kd_after_tax": kd_after_tax,
        "market_cap": market_cap,
        "we": we,
        "wd": wd,
    }
    return wacc, raw, True


def _compute_fcf_yield(cf, info):
    ocf = get_financial_item(cf, _KW_OPERATING_CASH_FLOW)
    capex = get_financial_item(cf, _KW_CAPEX)
    market_cap = info.get("marketCap")

    if ocf is None or capex is None or market_cap is None or market_cap == 0:
        return None, None, {}

    fcf = ocf - capex
    fcf_yield = fcf / market_cap
    raw = {"ocf": ocf, "capex": capex, "fcf": fcf, "market_cap": market_cap}
    return fcf, fcf_yield, raw


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


def calculate_value_score(ticker: str) -> dict:
    """
    Compute S2 (value creation, 0-100) for a ticker from the ROIC-WACC spread
    (60%) and FCF yield (40%).

    If WACC can't be computed from debt/interest data, falls back to a
    0.09 industry-median estimate and sets estimated_wacc=True. If one of
    the two sub-scores is missing, the other alone determines S2.
    """
    data = fetch_stock_data(ticker)
    bs = data.get("balance_sheet")
    fin = data.get("financials")
    cf = data.get("cashflow")
    info = data.get("info") or {}

    tax_rate, tax_rate_computed = _compute_tax_rate(fin)

    roic, roic_raw = _compute_roic(bs, fin, tax_rate)
    wacc, wacc_raw, wacc_computed = _compute_wacc(bs, fin, info, tax_rate)

    estimated_wacc = not wacc_computed
    if wacc is None:
        wacc = _ESTIMATED_WACC

    spread = roic - wacc if roic is not None else None

    fcf, fcf_yield, fcf_raw = _compute_fcf_yield(cf, info)

    eva = None
    if roic is not None and spread is not None and roic_raw.get("invested_capital") is not None:
        eva = spread * roic_raw["invested_capital"]

    spread_score = _map_spread(spread)
    fcf_yield_score = _map_fcf_yield(fcf_yield)

    weighted = {"spread": (0.60, spread_score), "fcf": (0.40, fcf_yield_score)}
    available = {key: (weight, score) for key, (weight, score) in weighted.items() if score is not None}

    if available:
        total_weight = sum(weight for weight, _ in available.values())
        s2 = sum((weight / total_weight) * score for weight, score in available.values())
    else:
        s2 = None

    details = {
        "roic": roic_raw,
        "wacc": wacc_raw,
        "fcf": fcf_raw,
        "tax_rate": tax_rate,
        "tax_rate_computed": tax_rate_computed,
        "spread_score": spread_score,
        "fcf_yield_score": fcf_yield_score,
        "sub_scores_computed": sorted(available.keys()),
    }

    return {
        "S2": round(s2, 2) if s2 is not None else None,
        "roic": round(roic, 4) if roic is not None else None,
        "wacc": round(wacc, 4) if wacc is not None else None,
        "spread": round(spread, 4) if spread is not None else None,
        "fcf_yield": round(fcf_yield, 4) if fcf_yield is not None else None,
        "eva": round(eva, 2) if eva is not None else None,
        "estimated_wacc": estimated_wacc,
        "details": details,
    }
