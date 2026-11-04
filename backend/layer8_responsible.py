"""
Layer 8 (S8): Operating & Financial Leverage.

Replaces an earlier ESG-based L8: yfinance's Sustainalytics ESG endpoint is
confirmed empty for essentially every ticker (Yahoo deprecated it), so that
version always fell through to a neutral-50 stub in practice -- not real
data. DOL/DFL/DTL measure a genuinely distinct risk dimension (earnings
sensitivity to revenue swings) that nothing else in the 9 layers captures,
and need no new data beyond what L1/L2 already fetch.

  - DOL (Degree of Operating Leverage) = %change EBIT / %change Revenue
  - DFL (Degree of Financial Leverage) = %change EPS / %change EBIT
  - DTL (Degree of Total Leverage)     = DOL x DFL

Lower magnitude = more stable earnings relative to revenue = safer = higher
score (same convention as L4's risk scoring). The %change/%change shape is
numerically unstable when a denominator is near zero or flips sign, so each
ratio is skipped (not computed) rather than reported when that happens.
"""

from .data_fetcher import fetch_fundamentals_annual, _linear_score, weighted_average

_MIN_PCT_CHANGE = 0.01  # below this, %change denominators are too noisy to divide by


def _year_fy(fundamentals, year):
    if year is None:
        if not fundamentals:
            return {}, {}, None
        year = max(fundamentals.keys())
    return fundamentals.get(year, {}), fundamentals.get(year - 1, {}), year


def _pct_change(current, prior):
    if current is None or not prior:
        return None
    change = (current - prior) / abs(prior)
    return change


def calculate_leverage_score(ticker: str, year: int | None = None) -> dict:
    """Compute S8 (0-100) from DOL (30%), DFL (30%), DTL (40%) -- the
    composite DTL gets the most weight since it's what ultimately matters
    (how much EPS swings per unit of revenue change), DOL/DFL each explain
    one half of that swing. Works identically for the live Analyze page
    (year=None) and the Backtest engine (an explicit year)."""
    fundamentals = fetch_fundamentals_annual(ticker)
    fy, prev_fy, resolved_year = _year_fy(fundamentals, year)

    revenue, prev_revenue = fy.get("revenue"), prev_fy.get("revenue")
    ebit, prev_ebit = fy.get("ebit"), prev_fy.get("ebit")
    shares, prev_shares = fy.get("shares_outstanding"), prev_fy.get("shares_outstanding")
    net_income, prev_net_income = fy.get("net_income"), prev_fy.get("net_income")
    eps = (net_income / shares) if (net_income is not None and shares) else None
    prev_eps = (prev_net_income / prev_shares) if (prev_net_income is not None and prev_shares) else None

    rev_pct = _pct_change(revenue, prev_revenue)
    ebit_pct = _pct_change(ebit, prev_ebit)
    eps_pct = _pct_change(eps, prev_eps)

    dol = None
    if rev_pct is not None and ebit_pct is not None and abs(rev_pct) >= _MIN_PCT_CHANGE:
        dol = ebit_pct / rev_pct

    dfl = None
    if ebit_pct is not None and eps_pct is not None and abs(ebit_pct) >= _MIN_PCT_CHANGE:
        dfl = eps_pct / ebit_pct

    dtl = None
    if rev_pct is not None and eps_pct is not None and abs(rev_pct) >= _MIN_PCT_CHANGE:
        dtl = eps_pct / rev_pct
    elif dol is not None and dfl is not None:
        dtl = dol * dfl

    # Lower |leverage| = earnings move less violently than revenue = safer.
    dol_score = _linear_score(abs(dol), good=1.5, bad=5.0, higher_is_better=False) if dol is not None else None
    dfl_score = _linear_score(abs(dfl), good=1.2, bad=3.0, higher_is_better=False) if dfl is not None else None
    dtl_score = _linear_score(abs(dtl), good=2.0, bad=8.0, higher_is_better=False) if dtl is not None else None

    s8 = weighted_average([(dol_score, 0.30), (dfl_score, 0.30), (dtl_score, 0.40)])

    if s8 is None:
        s8 = 50.0
        data_quality = "stub"
        details = ["Insufficient year-over-year fundamentals to compute operating/financial leverage"]
    else:
        data_quality = "full" if (dol is not None and dfl is not None) else "partial"
        details = []
        if dol is not None:
            details.append(f"DOL {dol:+.2f} -- EBIT moves {abs(dol):.1f}x as fast as revenue")
        if dfl is not None:
            details.append(f"DFL {dfl:+.2f} -- EPS moves {abs(dfl):.1f}x as fast as EBIT")
        if dtl is not None:
            details.append(f"DTL {dtl:+.2f} -- EPS moves {abs(dtl):.1f}x as fast as revenue overall")

    return {
        "S8": round(s8, 1),
        "dol": round(dol, 3) if dol is not None else None,
        "dfl": round(dfl, 3) if dfl is not None else None,
        "dtl": round(dtl, 3) if dtl is not None else None,
        "data_quality": data_quality,
        "details": details,
        "year": resolved_year,
    }


# Backwards-compatible name (app.py / prism_score.py historically called
# compute_layer8).
compute_layer8 = calculate_leverage_score
