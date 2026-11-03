"""
Layer 1 (S1): Fundamental Quality.

Combines three independent accounting-quality signals into a single 0-100 score:
  - Altman Z'' Score (emerging-markets variant) — bankruptcy / distress risk
  - Piotroski F-Score — fundamental strength trend (9 binary signals)
  - Beneish M-Score — earnings-manipulation risk

Any sub-score that cannot be computed because of missing data is dropped and
the remaining weights are renormalized to sum to 1.0 (see calculate_quality_score).
"""

import pandas as pd

from .data_fetcher import fetch_stock_data, get_financial_item

# ---------------------------------------------------------------------------
# Row-label keyword groups (yfinance naming varies by stock/region/version).
# ---------------------------------------------------------------------------
_KW_TOTAL_ASSETS = ["Total Assets"]
_KW_CURRENT_ASSETS = ["Total Current Assets", "Current Assets"]
_KW_CURRENT_LIABILITIES = ["Total Current Liabilities", "Current Liabilities"]
_KW_RETAINED_EARNINGS = ["Retained Earnings"]
_KW_TOTAL_LIABILITIES = ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"]
_KW_BOOK_EQUITY = [
    "Total Equity Gross Minority Interest",
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Common Stock Equity",
]
_KW_LONG_TERM_DEBT = ["Long Term Debt", "LongTermDebt"]
_KW_NET_RECEIVABLES = ["Net Receivables", "Accounts Receivable", "Receivables"]
_KW_NET_PPE = ["Net PPE", "Property Plant Equipment Net", "Properties Plant And Equipment Net"]
_KW_SHARES_OUTSTANDING = ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]

_KW_EBIT = ["EBIT"]
_KW_OPERATING_INCOME = ["Operating Income"]
_KW_NET_INCOME = ["Net Income Common Stockholders", "Net Income"]
_KW_TOTAL_REVENUE = ["Total Revenue", "Operating Revenue"]
_KW_COST_OF_REVENUE = ["Cost Of Revenue", "Reconciled Cost Of Revenue"]

_KW_OPERATING_CASH_FLOW = [
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]
_KW_DEPRECIATION = ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"]


# ---------------------------------------------------------------------------
# Lookup helpers. get_financial_item() (from data_fetcher) is used for every
# current-period (t) value, per spec. These local helpers reuse the exact
# same row-matching rule to additionally pull the prior-period (t-1) value,
# which get_financial_item does not expose.
# ---------------------------------------------------------------------------
def _get_row_values(df, keywords):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    normalized_keywords = [kw.lower().replace(" ", "") for kw in keywords]
    for row_label in df.index:
        row_norm = str(row_label).lower().replace(" ", "")
        for kw in normalized_keywords:
            if kw in row_norm or row_norm in kw:
                values = df.loc[row_label].dropna()
                if values.empty:
                    continue
                return values
    return None


def _prior(df, keywords):
    """Value from the period immediately before the most recent one (t-1)."""
    values = _get_row_values(df, keywords)
    if values is None or len(values) < 2:
        return None
    try:
        return float(values.iloc[1])
    except (TypeError, ValueError):
        return None


def _lookup_current(df, *keyword_groups):
    for keywords in keyword_groups:
        value = get_financial_item(df, keywords)
        if value is not None:
            return value
    return None


def _lookup_prior(df, *keyword_groups):
    for keywords in keyword_groups:
        value = _prior(df, keywords)
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# 1. Altman Z'' Score (emerging markets variant, no X5 industry term)
# ---------------------------------------------------------------------------
def _compute_altman_z(bs, fin):
    total_assets = get_financial_item(bs, _KW_TOTAL_ASSETS)
    current_assets = get_financial_item(bs, _KW_CURRENT_ASSETS)
    current_liabilities = get_financial_item(bs, _KW_CURRENT_LIABILITIES)
    retained_earnings = get_financial_item(bs, _KW_RETAINED_EARNINGS)
    ebit = _lookup_current(fin, _KW_EBIT, _KW_OPERATING_INCOME)
    total_liabilities = get_financial_item(bs, _KW_TOTAL_LIABILITIES)
    book_equity = get_financial_item(bs, _KW_BOOK_EQUITY)

    required = [
        total_assets,
        current_assets,
        current_liabilities,
        retained_earnings,
        ebit,
        total_liabilities,
        book_equity,
    ]
    if any(v is None for v in required) or total_assets == 0 or total_liabilities == 0:
        return None, {}

    working_capital = current_assets - current_liabilities
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities

    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    raw = {
        "total_assets": total_assets,
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "working_capital": working_capital,
        "retained_earnings": retained_earnings,
        "ebit": ebit,
        "total_liabilities": total_liabilities,
        "book_equity": book_equity,
        "x1_working_capital_to_assets": x1,
        "x2_retained_earnings_to_assets": x2,
        "x3_ebit_to_assets": x3,
        "x4_equity_to_liabilities": x4,
    }
    return z, raw


# ---------------------------------------------------------------------------
# 2. Piotroski F-Score (9 binary signals)
# ---------------------------------------------------------------------------
def _compute_piotroski_f(bs, fin, cf):
    ta_t = get_financial_item(bs, _KW_TOTAL_ASSETS)
    ta_p = _prior(bs, _KW_TOTAL_ASSETS)
    ni_t = get_financial_item(fin, _KW_NET_INCOME)
    ni_p = _prior(fin, _KW_NET_INCOME)
    cfo_t = get_financial_item(cf, _KW_OPERATING_CASH_FLOW)
    ltd_t = get_financial_item(bs, _KW_LONG_TERM_DEBT)
    ltd_p = _prior(bs, _KW_LONG_TERM_DEBT)
    ca_t = get_financial_item(bs, _KW_CURRENT_ASSETS)
    ca_p = _prior(bs, _KW_CURRENT_ASSETS)
    cl_t = get_financial_item(bs, _KW_CURRENT_LIABILITIES)
    cl_p = _prior(bs, _KW_CURRENT_LIABILITIES)
    shares_t = get_financial_item(bs, _KW_SHARES_OUTSTANDING)
    shares_p = _prior(bs, _KW_SHARES_OUTSTANDING)
    sales_t = get_financial_item(fin, _KW_TOTAL_REVENUE)
    sales_p = _prior(fin, _KW_TOTAL_REVENUE)
    cogs_t = get_financial_item(fin, _KW_COST_OF_REVENUE)
    cogs_p = _prior(fin, _KW_COST_OF_REVENUE)

    required = [
        ta_t, ta_p, ni_t, ni_p, cfo_t, ltd_t, ltd_p, ca_t, ca_p, cl_t, cl_p,
        shares_t, shares_p, sales_t, sales_p, cogs_t, cogs_p,
    ]
    if any(v is None for v in required):
        return None, {}
    if ta_t == 0 or ta_p == 0 or cl_t == 0 or cl_p == 0 or sales_t == 0 or sales_p == 0:
        return None, {}

    roa_t = ni_t / ta_t
    roa_p = ni_p / ta_p
    ltd_ratio_t = ltd_t / ta_t
    ltd_ratio_p = ltd_p / ta_p
    current_ratio_t = ca_t / cl_t
    current_ratio_p = ca_p / cl_p
    gross_margin_t = (sales_t - cogs_t) / sales_t
    gross_margin_p = (sales_p - cogs_p) / sales_p
    asset_turnover_t = sales_t / ta_t
    asset_turnover_p = sales_p / ta_p

    signals = {
        "P1_roa_positive": roa_t > 0,
        "P2_cfo_positive": cfo_t > 0,
        "P3_delta_roa_positive": roa_t > roa_p,
        "P4_accrual_cfo_exceeds_roa": (cfo_t / ta_t) > roa_t,
        "P5_delta_leverage_down": ltd_ratio_t < ltd_ratio_p,
        "P6_delta_current_ratio_up": current_ratio_t > current_ratio_p,
        "P7_no_new_shares": shares_t <= shares_p,
        "P8_delta_gross_margin_up": gross_margin_t > gross_margin_p,
        "P9_delta_asset_turnover_up": asset_turnover_t > asset_turnover_p,
    }

    f = sum(1 for signal in signals.values() if signal)

    raw = {
        "roa_t": roa_t,
        "roa_prior": roa_p,
        "cfo_t": cfo_t,
        "long_term_debt_ratio_t": ltd_ratio_t,
        "long_term_debt_ratio_prior": ltd_ratio_p,
        "current_ratio_t": current_ratio_t,
        "current_ratio_prior": current_ratio_p,
        "shares_outstanding_t": shares_t,
        "shares_outstanding_prior": shares_p,
        "gross_margin_t": gross_margin_t,
        "gross_margin_prior": gross_margin_p,
        "asset_turnover_t": asset_turnover_t,
        "asset_turnover_prior": asset_turnover_p,
        "signals": signals,
    }
    return f, raw


# ---------------------------------------------------------------------------
# 3. Beneish M-Score (5-variable model)
# ---------------------------------------------------------------------------
def _compute_beneish_m(bs, fin, cf):
    recv_t = get_financial_item(bs, _KW_NET_RECEIVABLES)
    recv_p = _prior(bs, _KW_NET_RECEIVABLES)
    sales_t = get_financial_item(fin, _KW_TOTAL_REVENUE)
    sales_p = _prior(fin, _KW_TOTAL_REVENUE)
    cogs_t = get_financial_item(fin, _KW_COST_OF_REVENUE)
    cogs_p = _prior(fin, _KW_COST_OF_REVENUE)
    ca_t = get_financial_item(bs, _KW_CURRENT_ASSETS)
    ca_p = _prior(bs, _KW_CURRENT_ASSETS)
    ppe_t = get_financial_item(bs, _KW_NET_PPE)
    ppe_p = _prior(bs, _KW_NET_PPE)
    ta_t = get_financial_item(bs, _KW_TOTAL_ASSETS)
    ta_p = _prior(bs, _KW_TOTAL_ASSETS)
    dep_t = get_financial_item(cf, _KW_DEPRECIATION)
    dep_p = _prior(cf, _KW_DEPRECIATION)

    required = [
        recv_t, recv_p, sales_t, sales_p, cogs_t, cogs_p, ca_t, ca_p,
        ppe_t, ppe_p, ta_t, ta_p, dep_t, dep_p,
    ]
    if any(v is None for v in required):
        return None, {}
    if sales_t == 0 or sales_p == 0 or ta_t == 0 or ta_p == 0:
        return None, {}

    gm_t = (sales_t - cogs_t) / sales_t
    gm_p = (sales_p - cogs_p) / sales_p
    if gm_t == 0:
        return None, {}

    aqi_t = 1 - (ca_t + ppe_t) / ta_t
    aqi_p = 1 - (ca_p + ppe_p) / ta_p
    depi_denom_t = ppe_t + dep_t
    depi_denom_p = ppe_p + dep_p
    if aqi_p == 0 or depi_denom_t == 0 or depi_denom_p == 0 or recv_p == 0 or dep_t == 0:
        return None, {}

    try:
        dsri = (recv_t / sales_t) / (recv_p / sales_p)
        gmi = gm_p / gm_t
        aqi = aqi_t / aqi_p
        sgi = sales_t / sales_p
        depi = (dep_p / depi_denom_p) / (dep_t / depi_denom_t)
    except ZeroDivisionError:
        return None, {}

    m = -6.065 + 0.823 * dsri + 0.906 * gmi + 0.593 * aqi + 0.717 * sgi + 0.107 * depi

    raw = {
        "dsri": dsri,
        "gmi": gmi,
        "aqi": aqi,
        "sgi": sgi,
        "depi": depi,
        "gross_margin_t": gm_t,
        "gross_margin_prior": gm_p,
    }
    return m, raw


# ---------------------------------------------------------------------------
# Score mappings (0-100)
# ---------------------------------------------------------------------------
def _map_altman(z):
    if z is None:
        return None
    if z >= 2.6:
        return 100.0
    if z >= 1.1:
        return 50.0
    return 0.0


def _map_piotroski(f):
    if f is None:
        return None
    if f >= 7:
        return 100.0
    if f >= 4:
        return 60.0
    return 20.0


def _map_beneish(m):
    if m is None:
        return None
    if m < -2.22:
        return 100.0
    return 20.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def calculate_quality_score(ticker: str) -> dict:
    """
    Compute S1 (fundamental quality, 0-100) for a ticker from Altman Z'',
    Piotroski F-Score, and Beneish M-Score, weighted 40/40/20.

    Any sub-score that can't be computed (missing data) is dropped and the
    remaining weights are renormalized to sum to 1.0.
    """
    data = fetch_stock_data(ticker)
    bs = data.get("balance_sheet")
    fin = data.get("financials")
    cf = data.get("cashflow")

    altman_z, altman_raw = _compute_altman_z(bs, fin)
    piotroski_f, piotroski_raw = _compute_piotroski_f(bs, fin, cf)
    beneish_m, beneish_raw = _compute_beneish_m(bs, fin, cf)

    z_score = _map_altman(altman_z)
    f_score = _map_piotroski(piotroski_f)
    m_score = _map_beneish(beneish_m)

    weighted = {"z": (0.40, z_score), "f": (0.40, f_score), "m": (0.20, m_score)}
    available = {key: (weight, score) for key, (weight, score) in weighted.items() if score is not None}

    computed_count = len(available)
    if computed_count == 0:
        s1 = 0.0
        data_quality = "insufficient"
        weights_used = {}
    else:
        total_weight = sum(weight for weight, _ in available.values())
        s1 = sum((weight / total_weight) * score for weight, score in available.values())
        weights_used = {key: round(weight / total_weight, 4) for key, (weight, _) in available.items()}
        data_quality = "full" if computed_count == 3 else "partial"

    details = {
        "altman": altman_raw,
        "piotroski": piotroski_raw,
        "beneish": beneish_raw,
        "sub_scores_computed": sorted(available.keys()),
        "weights_used": weights_used,
    }

    return {
        "S1": round(s1, 2),
        "altman_z": round(altman_z, 4) if altman_z is not None else None,
        "piotroski_f": piotroski_f,
        "beneish_m": round(beneish_m, 4) if beneish_m is not None else None,
        "z_score": z_score,
        "f_score": f_score,
        "m_score": m_score,
        "details": details,
        "data_quality": data_quality,
    }
