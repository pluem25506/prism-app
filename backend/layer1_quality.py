"""
Layer 1 (S1): Fundamental Quality.

Combines four independent accounting-quality signals into a single 0-100
score:
  - Altman Z'' Score (emerging-markets variant) -- bankruptcy/distress risk
  - Piotroski F-Score -- fundamental strength trend (9 binary signals)
  - Beneish M-Score -- earnings-manipulation risk
  - DPS Growth -- dividend-per-share trend, a shareholder-friendliness signal

Works for both the live Analyze page (year=None -> latest available fiscal
year) and the Backtest engine (an explicit historical year), since both now
read from the same year-indexed fundamentals dict (fetch_fundamentals_annual)
instead of always reading column 0 of a fresh yfinance DataFrame.

Any sub-score that cannot be computed because of missing data is dropped and
the remaining weights are renormalized to sum to 1.0.
"""

from .data_fetcher import fetch_fundamentals_annual, fetch_dividends, _linear_score


def _year_fy(fundamentals: dict, year):
    if year is None:
        if not fundamentals:
            return {}, {}, None
        year = max(fundamentals.keys())
    return fundamentals.get(year, {}), fundamentals.get(year - 1, {}), year


# ---------------------------------------------------------------------------
# 1. Altman Z'' Score (emerging markets variant, no X5 industry term)
# ---------------------------------------------------------------------------
def _compute_altman_z(fy):
    total_assets = fy.get("total_assets")
    current_assets = fy.get("current_assets")
    current_liabilities = fy.get("current_liabilities")
    retained_earnings = fy.get("retained_earnings")
    ebit = fy.get("ebit")
    total_liabilities = fy.get("total_liabilities")
    book_equity = fy.get("total_equity")

    required = [total_assets, current_assets, current_liabilities, retained_earnings, ebit, total_liabilities, book_equity]
    if any(v is None for v in required) or not total_assets or not total_liabilities:
        return None, {}

    working_capital = current_assets - current_liabilities
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities
    z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    raw = {
        "working_capital": working_capital, "retained_earnings": retained_earnings, "ebit": ebit,
        "total_liabilities": total_liabilities, "book_equity": book_equity,
        "x1_working_capital_to_assets": x1, "x2_retained_earnings_to_assets": x2,
        "x3_ebit_to_assets": x3, "x4_equity_to_liabilities": x4,
    }
    return z, raw


# ---------------------------------------------------------------------------
# 2. Piotroski F-Score (9 binary signals)
# ---------------------------------------------------------------------------
def _compute_piotroski_f(fy, prev_fy):
    ta_t, ta_p = fy.get("total_assets"), prev_fy.get("total_assets")
    ni_t, ni_p = fy.get("net_income"), prev_fy.get("net_income")
    cfo_t = fy.get("operating_cashflow")
    ltd_t, ltd_p = fy.get("long_term_debt"), prev_fy.get("long_term_debt")
    ca_t, ca_p = fy.get("current_assets"), prev_fy.get("current_assets")
    cl_t, cl_p = fy.get("current_liabilities"), prev_fy.get("current_liabilities")
    shares_t, shares_p = fy.get("shares_outstanding"), prev_fy.get("shares_outstanding")
    sales_t, sales_p = fy.get("revenue"), prev_fy.get("revenue")
    cogs_t, cogs_p = fy.get("cost_of_revenue"), prev_fy.get("cost_of_revenue")

    required = [ta_t, ta_p, ni_t, ni_p, cfo_t, ltd_t, ltd_p, ca_t, ca_p, cl_t, cl_p, shares_t, shares_p, sales_t, sales_p, cogs_t, cogs_p]
    if any(v is None for v in required) or not ta_t or not ta_p or not cl_t or not cl_p or not sales_t or not sales_p:
        return None, {}

    roa_t, roa_p = ni_t / ta_t, ni_p / ta_p
    ltd_ratio_t, ltd_ratio_p = ltd_t / ta_t, ltd_p / ta_p
    current_ratio_t, current_ratio_p = ca_t / cl_t, ca_p / cl_p
    gross_margin_t = (sales_t - cogs_t) / sales_t
    gross_margin_p = (sales_p - cogs_p) / sales_p
    asset_turnover_t, asset_turnover_p = sales_t / ta_t, sales_p / ta_p

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
    f = sum(1 for v in signals.values() if v)
    raw = {"roa_t": roa_t, "roa_prior": roa_p, "cfo_t": cfo_t, "signals": signals}
    return f, raw


# ---------------------------------------------------------------------------
# 3. Beneish M-Score (5-variable model)
# ---------------------------------------------------------------------------
def _compute_beneish_m(fy, prev_fy):
    recv_t, recv_p = fy.get("accounts_receivable"), prev_fy.get("accounts_receivable")
    sales_t, sales_p = fy.get("revenue"), prev_fy.get("revenue")
    cogs_t, cogs_p = fy.get("cost_of_revenue"), prev_fy.get("cost_of_revenue")
    ca_t, ca_p = fy.get("current_assets"), prev_fy.get("current_assets")
    ppe_t, ppe_p = fy.get("net_ppe"), prev_fy.get("net_ppe")
    ta_t, ta_p = fy.get("total_assets"), prev_fy.get("total_assets")
    dep_t, dep_p = fy.get("depreciation"), prev_fy.get("depreciation")

    required = [recv_t, recv_p, sales_t, sales_p, cogs_t, cogs_p, ca_t, ca_p, ppe_t, ppe_p, ta_t, ta_p, dep_t, dep_p]
    if any(v is None for v in required) or not sales_t or not sales_p or not ta_t or not ta_p:
        return None, {}

    gm_t = (sales_t - cogs_t) / sales_t
    gm_p = (sales_p - cogs_p) / sales_p
    if gm_t == 0:
        return None, {}

    aqi_t = 1 - (ca_t + ppe_t) / ta_t
    aqi_p = 1 - (ca_p + ppe_p) / ta_p
    depi_denom_t = ppe_t + dep_t
    depi_denom_p = ppe_p + dep_p
    if aqi_p == 0 or depi_denom_t == 0 or depi_denom_p == 0 or not recv_p or not dep_t:
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
    return m, {"dsri": dsri, "gmi": gmi, "aqi": aqi, "sgi": sgi, "depi": depi}


# ---------------------------------------------------------------------------
# 4. DPS Growth (YoY)
# ---------------------------------------------------------------------------
def _compute_dps_growth_score(ticker, year):
    try:
        dividends = fetch_dividends(ticker)
    except Exception:
        return None
    if dividends is None or dividends.empty or year is None:
        return None
    this_year_divs = dividends[dividends.index.year == year]
    prev_year_divs = dividends[dividends.index.year == year - 1]
    if this_year_divs.empty or prev_year_divs.empty:
        return None
    this_total = float(this_year_divs.sum())
    prev_total = float(prev_year_divs.sum())
    if not prev_total:
        return None
    dps_growth = (this_total - prev_total) / abs(prev_total)
    return _linear_score(dps_growth, good=0.10, bad=-0.20, higher_is_better=True)


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
def calculate_quality_score(ticker: str, year: int | None = None) -> dict:
    """Compute S1 (fundamental quality, 0-100) from Altman Z'' (35%),
    Piotroski F-Score (35%), Beneish M-Score (15%), and DPS Growth (15%).

    year=None (the live Analyze page's usage) scores the latest fiscal year
    yfinance has fundamentals for. An explicit year (the Backtest engine's
    usage) scores that historical fiscal year instead, from the same
    fetch_fundamentals_annual() dict -- so both pages run the exact same
    formula, just pointed at a different year.

    Any sub-score that can't be computed (missing data) is dropped and the
    remaining weights are renormalized to sum to 1.0.
    """
    fundamentals = fetch_fundamentals_annual(ticker)
    fy, prev_fy, resolved_year = _year_fy(fundamentals, year)

    altman_z, altman_raw = _compute_altman_z(fy)
    piotroski_f, piotroski_raw = _compute_piotroski_f(fy, prev_fy)
    beneish_m, beneish_raw = _compute_beneish_m(fy, prev_fy)
    dps_growth_score = _compute_dps_growth_score(ticker, resolved_year)

    z_score = _map_altman(altman_z)
    f_score = _map_piotroski(piotroski_f)
    m_score = _map_beneish(beneish_m)

    weighted = {"z": (0.35, z_score), "f": (0.35, f_score), "m": (0.15, m_score), "dps_growth": (0.15, dps_growth_score)}
    available = {key: (weight, score) for key, (weight, score) in weighted.items() if score is not None}

    if not available:
        # None (not 0.0) -- a ticker with zero computable sub-scores (e.g.
        # crypto, which has no fundamentals at all) should drop out of the
        # composite via the usual weight-renormalization, not drag the
        # whole PRISM score down with an artificial failing grade.
        s1 = None
        data_quality = "insufficient"
        weights_used = {}
    else:
        total_weight = sum(weight for weight, _ in available.values())
        s1 = sum((weight / total_weight) * score for weight, score in available.values())
        weights_used = {key: round(weight / total_weight, 4) for key, (weight, _) in available.items()}
        data_quality = "full" if len(available) == 4 else "partial"

    details = {
        "altman": altman_raw, "piotroski": piotroski_raw, "beneish": beneish_raw,
        "sub_scores_computed": sorted(available.keys()), "weights_used": weights_used,
        "year": resolved_year,
    }

    return {
        "S1": round(s1, 2) if s1 is not None else None,
        "altman_z": round(altman_z, 4) if altman_z is not None else None,
        "piotroski_f": piotroski_f,
        "beneish_m": round(beneish_m, 4) if beneish_m is not None else None,
        "dps_growth_score": dps_growth_score,
        "z_score": z_score, "f_score": f_score, "m_score": m_score,
        "details": details,
        "data_quality": data_quality,
    }
