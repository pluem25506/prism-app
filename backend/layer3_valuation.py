"""
Layer 3 (S3): Valuation.

Nine independent valuation models each cast a BUY / HOLD / SELL / ABSTAIN
vote. The final verdict is decided by majority vote (see
calculate_valuation_score).

Rebuilt to compute every ratio from fundamentals + price (via the shared
fetch_fundamentals_annual() dict) instead of yfinance's live-only `info`
snapshot, so the exact same code scores either today (year=None, the live
Analyze page) or a historical fiscal year (the Backtest engine). Two real
bugs surfaced and were fixed in the process:
  - P/E's "sector_pe" comparison read info["sectorPE"], a key yfinance never
    actually populates; the code defaulted it to `pe` itself, so the vote
    was comparing pe against pe*0.85/pe*1.15 -- always HOLD, for every
    ticker, permanently. Replaced with fixed cheap/expensive anchors.
  - P/FCF's "fcf = ocf - capex" subtracted a negative (yfinance reports
    capex as a cash outflow, i.e. negative) instead of adding it, inflating
    FCF. Fixed to "ocf + capex" (same fix as layer2_value.py's FCF Yield).
"""

import math

from .data_fetcher import fetch_fundamentals_annual, fetch_price_history_long


def _year_fy(fundamentals, year):
    if year is None:
        if not fundamentals:
            return {}, {}, None
        year = max(fundamentals.keys())
    return fundamentals.get(year, {}), fundamentals.get(year - 1, {}), year


def _eps_bvps(fy):
    shares = fy.get("shares_outstanding")
    net_income = fy.get("net_income")
    equity = fy.get("total_equity")
    eps = (net_income / shares) if (net_income is not None and shares) else None
    bvps = (equity / shares) if (equity is not None and shares) else None
    return eps, bvps


def _eps_growth(fy, prev_fy):
    net_income, shares = fy.get("net_income"), fy.get("shares_outstanding")
    prev_net_income, prev_shares = prev_fy.get("net_income"), prev_fy.get("shares_outstanding")
    eps = (net_income / shares) if (net_income is not None and shares) else None
    prev_eps = (prev_net_income / prev_shares) if (prev_net_income is not None and prev_shares) else None
    if eps is None or not prev_eps:
        return None
    return (eps - prev_eps) / abs(prev_eps)


# ---------------------------------------------------------------------------
# 1. P/E Ratio -- fixed cheap/expensive anchors (the live "sector_pe" field
#    never actually existed; see module docstring).
# ---------------------------------------------------------------------------
def _vote_pe(eps, price):
    if not eps or eps <= 0 or not price:
        return "ABSTAIN", {"pe": None}
    pe = price / eps
    if pe < 15:
        vote = "BUY"
    elif pe > 25:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"pe": pe}


# ---------------------------------------------------------------------------
# 2. EV/EBITDA
# ---------------------------------------------------------------------------
def _vote_ev_ebitda(fy, price, shares):
    ebitda = fy.get("ebitda")
    if not ebitda or ebitda <= 0 or not price or not shares:
        return "ABSTAIN", {"ev_ebitda": None}
    market_cap = price * shares
    ev = market_cap + (fy.get("total_debt") or 0) - (fy.get("cash") or 0)
    ev_ebitda = ev / ebitda
    if ev_ebitda < 7:
        vote = "BUY"
    elif ev_ebitda > 12:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"ev_ebitda": ev_ebitda}


# ---------------------------------------------------------------------------
# 3. P/FCF
# ---------------------------------------------------------------------------
def _vote_p_fcf(fy, price, shares):
    ocf, capex = fy.get("operating_cashflow"), fy.get("capex")
    if ocf is None or capex is None or not price or not shares:
        return "ABSTAIN", {"fcf": None, "p_fcf": None}
    fcf = ocf + capex  # capex is already negative (cash outflow)
    if fcf <= 0:
        return "ABSTAIN", {"fcf": fcf, "p_fcf": None}
    p_fcf = (price * shares) / fcf
    if p_fcf < 10:
        vote = "BUY"
    elif p_fcf > 20:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"fcf": fcf, "p_fcf": p_fcf}


# ---------------------------------------------------------------------------
# 4. PEG Ratio -- growth uses trailing EPS YoY (no historical analogue of
#    yfinance's forward-looking live "earningsGrowth" consensus estimate).
# ---------------------------------------------------------------------------
def _vote_peg(eps, price, eps_growth):
    if not eps or eps <= 0 or not price or not eps_growth or eps_growth <= 0:
        return "ABSTAIN", {"peg": None}
    pe = price / eps
    peg = pe / (eps_growth * 100)
    if peg < 0.8:
        vote = "BUY"
    elif peg > 1.5:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"peg": peg}


# ---------------------------------------------------------------------------
# 5. Reverse DCF (implied growth rate)
# ---------------------------------------------------------------------------
def _vote_reverse_dcf(eps, price, eps_growth, wacc=0.09):
    if not price or not eps or (price + eps) == 0:
        return "ABSTAIN", {"implied_g": None, "historical_g": None}
    implied_g = (price * wacc - eps) / (price + eps)
    historical_g = eps_growth or 0
    if implied_g > historical_g * 1.5:
        vote = "SELL"
    elif implied_g < historical_g * 0.8:
        vote = "BUY"
    else:
        vote = "HOLD"
    return vote, {"implied_g": implied_g, "historical_g": historical_g}


# ---------------------------------------------------------------------------
# 6. Graham Number
# ---------------------------------------------------------------------------
def _vote_graham(eps, bvps, price):
    if not eps or eps <= 0 or not bvps or bvps <= 0 or not price:
        return "ABSTAIN", {"graham": None}
    graham = math.sqrt(22.5 * eps * bvps)
    if price < graham * 0.85:
        vote = "BUY"
    elif price > graham * 1.15:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"graham": graham}


# ---------------------------------------------------------------------------
# 7. DDM (conditional on the stock paying a dividend)
# ---------------------------------------------------------------------------
def _vote_ddm(fy, price, dividend_per_share, beta, div_growth):
    if not dividend_per_share or not price:
        return "ABSTAIN", {"div_yield": None, "intrinsic": None}
    div_yield = dividend_per_share / price
    ke = 0.028 + beta * 0.065
    if ke <= div_growth:
        return "ABSTAIN", {"div_yield": div_yield, "intrinsic": None}
    intrinsic = (div_yield * price * (1 + div_growth)) / (ke - div_growth)
    if price < intrinsic * 0.90:
        vote = "BUY"
    elif price > intrinsic * 1.10:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"div_yield": div_yield, "intrinsic": intrinsic}


# ---------------------------------------------------------------------------
# 8. P/BV -- sector-aware (sector classification is current/live even in
#    historical mode -- a company's sector rarely changes year to year, an
#    acceptable simplification since yfinance has no historical sector field)
# ---------------------------------------------------------------------------
def _vote_pbv(bvps, price, sector):
    if not bvps or bvps <= 0 or not price:
        return "ABSTAIN", {"pb": None}
    pb = price / bvps
    if "Financial" in sector or "Bank" in sector:
        vote = "BUY" if pb < 0.8 else "SELL" if pb > 1.8 else "HOLD"
    else:
        vote = "BUY" if pb < 1.5 else "SELL" if pb > 3.5 else "HOLD"
    return vote, {"pb": pb, "sector": sector}


# ---------------------------------------------------------------------------
# 9. P/S (Price to Sales) -- new
# ---------------------------------------------------------------------------
def _vote_ps(fy, price, shares):
    revenue = fy.get("revenue")
    if not revenue or revenue <= 0 or not price or not shares:
        return "ABSTAIN", {"ps": None}
    ps = (price * shares) / revenue
    if ps < 2:
        vote = "BUY"
    elif ps > 8:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"ps": ps}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def calculate_valuation_score(ticker: str, year: int | None = None) -> dict:
    """Compute S3 (valuation, 0-100) via majority vote across nine
    independent valuation models (P/E, EV/EBITDA, P/FCF, PEG, Reverse DCF,
    Graham Number, DDM, P/BV, P/S). Models that can't be computed ABSTAIN
    and are excluded from the vote tally.

    year=None (live) prices at today's close; an explicit year (Backtest)
    prices at that fiscal year's year-end close, reconstructing every ratio
    from fundamentals instead of yfinance's live-only `info` snapshot.
    """
    fundamentals = fetch_fundamentals_annual(ticker)
    fy, prev_fy, resolved_year = _year_fy(fundamentals, year)
    price_history = fetch_price_history_long(ticker)

    if price_history.empty:
        price = None
    elif year is None:
        price = float(price_history["Close"].iloc[-1])
    else:
        year_rows = price_history[price_history.index.year == year]
        price = float(year_rows["Close"].iloc[-1]) if not year_rows.empty else None

    shares = fy.get("shares_outstanding")
    eps, bvps = _eps_bvps(fy)
    eps_growth = _eps_growth(fy, prev_fy)
    sector = fy.get("sector", "") or ""

    # Dividend per share (for DDM) and its YoY growth (for DDM's growth term)
    dividend_per_share = None
    div_growth = 0.03
    try:
        from .data_fetcher import fetch_dividends
        dividends = fetch_dividends(ticker)
        if dividends is not None and not dividends.empty and resolved_year is not None:
            this_year = dividends[dividends.index.year == resolved_year]
            prev_year = dividends[dividends.index.year == resolved_year - 1]
            if not this_year.empty:
                dividend_per_share = float(this_year.sum())
            if not this_year.empty and not prev_year.empty and prev_year.sum():
                div_growth = (float(this_year.sum()) - float(prev_year.sum())) / abs(float(prev_year.sum()))
    except Exception:
        pass

    beta = 1.0  # DDM's cost-of-equity input; see layer2_value._historical_beta for the full computation

    pe_vote, pe_details = _vote_pe(eps, price)
    ev_vote, ev_details = _vote_ev_ebitda(fy, price, shares)
    fcf_vote, fcf_details = _vote_p_fcf(fy, price, shares)
    peg_vote, peg_details = _vote_peg(eps, price, eps_growth)
    dcf_vote, dcf_details = _vote_reverse_dcf(eps, price, eps_growth)
    graham_vote, graham_details = _vote_graham(eps, bvps, price)
    ddm_vote, ddm_details = _vote_ddm(fy, price, dividend_per_share, beta, div_growth)
    pbv_vote, pbv_details = _vote_pbv(bvps, price, sector)
    ps_vote, ps_details = _vote_ps(fy, price, shares)

    votes = {
        "PE": pe_vote, "EV_EBITDA": ev_vote, "P_FCF": fcf_vote, "PEG": peg_vote,
        "Reverse_DCF": dcf_vote, "Graham": graham_vote, "DDM": ddm_vote, "P_BV": pbv_vote, "P_S": ps_vote,
    }

    all_votes = list(votes.values())
    valid_votes = [v for v in all_votes if v != "ABSTAIN"]
    buy_count = valid_votes.count("BUY")
    hold_count = valid_votes.count("HOLD")
    sell_count = valid_votes.count("SELL")
    abstain_count = all_votes.count("ABSTAIN")
    n = len(valid_votes)

    insufficient_data = n == 0
    if insufficient_data:
        signal = "MIXED SIGNAL"
    elif buy_count / n >= 0.60:
        signal = "BUY"
    elif sell_count / n >= 0.60:
        signal = "SELL"
    elif buy_count > sell_count:
        signal = "BUY"
    elif sell_count > buy_count:
        signal = "SELL"
    elif buy_count == 0 and sell_count == 0:
        signal = "HOLD"
    else:
        signal = "MIXED SIGNAL"

    mixed_signal_flag = signal == "MIXED SIGNAL"

    if signal == "BUY":
        s3 = 75 + (buy_count / n) * 25
    elif signal == "SELL":
        s3 = 25 - (sell_count / n) * 25
    else:
        s3 = 50.0

    details = {
        "PE": pe_details, "EV_EBITDA": ev_details, "P_FCF": fcf_details, "PEG": peg_details,
        "Reverse_DCF": dcf_details, "Graham": graham_details, "DDM": ddm_details,
        "P_BV": pbv_details, "P_S": ps_details, "insufficient_data": insufficient_data,
        "year": resolved_year,
    }

    return {
        "S3": round(s3, 2),
        "signal": signal,
        "mixed_signal_flag": mixed_signal_flag,
        "votes": votes,
        "buy_count": buy_count, "hold_count": hold_count, "sell_count": sell_count, "abstain_count": abstain_count,
        "details": details,
    }
