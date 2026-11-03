"""
Layer 3 (S3): Valuation.

Eight independent valuation models each cast a BUY / HOLD / SELL / ABSTAIN vote.
The final verdict is decided by majority vote (see calculate_valuation_score).
"""

import math

from .data_fetcher import fetch_stock_data, get_financial_item, normalize_dividend_yield

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


# ---------------------------------------------------------------------------
# 1. P/E Ratio
# ---------------------------------------------------------------------------
def _vote_pe(info):
    pe = info.get("trailingPE")
    if pe is None or pe < 0:
        return "ABSTAIN", {"pe": pe, "sector_pe": None}

    sector_pe = info.get("sectorPE", pe)
    if sector_pe is None:
        sector_pe = pe

    if pe < sector_pe * 0.85:
        vote = "BUY"
    elif pe > sector_pe * 1.15:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"pe": pe, "sector_pe": sector_pe}


# ---------------------------------------------------------------------------
# 2. EV/EBITDA
# ---------------------------------------------------------------------------
def _vote_ev_ebitda(info):
    ev = info.get("enterpriseValue")
    ebitda = info.get("ebitda")
    if ev is None or ebitda is None or ebitda <= 0:
        return "ABSTAIN", {"ev": ev, "ebitda": ebitda, "ev_ebitda": None}

    ev_ebitda = ev / ebitda
    if ev_ebitda < 7:
        vote = "BUY"
    elif ev_ebitda > 12:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"ev": ev, "ebitda": ebitda, "ev_ebitda": ev_ebitda}


# ---------------------------------------------------------------------------
# 3. P/FCF
# ---------------------------------------------------------------------------
def _vote_p_fcf(info, cashflow):
    market_cap = info.get("marketCap")
    ocf = get_financial_item(cashflow, _KW_OPERATING_CASH_FLOW)
    capex = get_financial_item(cashflow, _KW_CAPEX)

    if market_cap is None or ocf is None or capex is None:
        return "ABSTAIN", {"market_cap": market_cap, "ocf": ocf, "capex": capex, "fcf": None, "p_fcf": None}

    fcf = ocf - capex
    if fcf <= 0:
        return "ABSTAIN", {"market_cap": market_cap, "ocf": ocf, "capex": capex, "fcf": fcf, "p_fcf": None}

    p_fcf = market_cap / fcf
    if p_fcf < 10:
        vote = "BUY"
    elif p_fcf > 20:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"market_cap": market_cap, "ocf": ocf, "capex": capex, "fcf": fcf, "p_fcf": p_fcf}


# ---------------------------------------------------------------------------
# 4. PEG Ratio
# ---------------------------------------------------------------------------
def _vote_peg(info):
    pe = info.get("trailingPE")
    growth = info.get("earningsGrowth", 0)
    if growth is None:
        growth = 0

    if pe is None or pe < 0 or growth <= 0:
        return "ABSTAIN", {"pe": pe, "growth": growth, "peg": None}

    peg = pe / (growth * 100)
    if peg < 0.8:
        vote = "BUY"
    elif peg > 1.5:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"pe": pe, "growth": growth, "peg": peg}


# ---------------------------------------------------------------------------
# 5. Reverse DCF (implied growth rate)
# ---------------------------------------------------------------------------
def _vote_reverse_dcf(info, wacc=0.09):
    price = info.get("currentPrice")
    eps = info.get("trailingEps")

    if price is None or eps is None or (price + eps) == 0:
        return "ABSTAIN", {"price": price, "eps": eps, "wacc": wacc, "implied_g": None, "historical_g": None}

    implied_g = (price * wacc - eps) / (price + eps)
    historical_g = info.get("earningsGrowth", 0)
    if historical_g is None:
        historical_g = 0

    if implied_g > historical_g * 1.5:
        vote = "SELL"
    elif implied_g < historical_g * 0.8:
        vote = "BUY"
    else:
        vote = "HOLD"
    return vote, {
        "price": price,
        "eps": eps,
        "wacc": wacc,
        "implied_g": implied_g,
        "historical_g": historical_g,
    }


# ---------------------------------------------------------------------------
# 6. Graham Number
# ---------------------------------------------------------------------------
def _vote_graham(info):
    eps = info.get("trailingEps")
    bvps = info.get("bookValue")
    price = info.get("currentPrice")

    if eps is None or bvps is None or price is None or eps <= 0 or bvps <= 0:
        return "ABSTAIN", {"eps": eps, "bvps": bvps, "price": price, "graham": None}

    graham = math.sqrt(22.5 * eps * bvps)
    if price < graham * 0.85:
        vote = "BUY"
    elif price > graham * 1.15:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {"eps": eps, "bvps": bvps, "price": price, "graham": graham}


# ---------------------------------------------------------------------------
# 7. DDM (conditional on the stock paying a dividend)
# ---------------------------------------------------------------------------
def _vote_ddm(info):
    price = info.get("currentPrice")
    div_yield = normalize_dividend_yield(info.get("dividendYield", 0), info.get("dividendRate"), price)
    if not div_yield:
        return "ABSTAIN", {"div_yield": div_yield, "intrinsic": None}

    beta = info.get("beta")
    if beta is None or price is None:
        return "ABSTAIN", {"div_yield": div_yield, "beta": beta, "price": price, "intrinsic": None}

    ke = 0.028 + beta * 0.065
    div_growth = info.get("dividendGrowthRate", 0.03)
    if div_growth is None:
        div_growth = 0.03

    if ke <= div_growth:
        return "ABSTAIN", {
            "div_yield": div_yield,
            "beta": beta,
            "ke": ke,
            "div_growth": div_growth,
            "price": price,
            "intrinsic": None,
        }

    intrinsic = (div_yield * price * (1 + div_growth)) / (ke - div_growth)
    if price < intrinsic * 0.90:
        vote = "BUY"
    elif price > intrinsic * 1.10:
        vote = "SELL"
    else:
        vote = "HOLD"
    return vote, {
        "div_yield": div_yield,
        "beta": beta,
        "ke": ke,
        "div_growth": div_growth,
        "price": price,
        "intrinsic": intrinsic,
    }


# ---------------------------------------------------------------------------
# 8. P/BV (Price to Book Value) -- sector-aware, especially relevant for
#    Thai bank stocks which routinely trade close to (or below) book value.
# ---------------------------------------------------------------------------
def _vote_pbv(info):
    bvps = info.get("bookValue")
    price = info.get("currentPrice")
    sector = info.get("sector", "") or ""

    if bvps is None or bvps <= 0 or price is None:
        return "ABSTAIN", {"bvps": bvps, "price": price, "pb": None, "sector": sector}

    pb = price / bvps

    if "Financial" in sector or "Bank" in sector:
        if pb < 0.8:
            vote = "BUY"
        elif pb > 1.8:
            vote = "SELL"
        else:
            vote = "HOLD"
    else:
        if pb < 1.5:
            vote = "BUY"
        elif pb > 3.5:
            vote = "SELL"
        else:
            vote = "HOLD"

    return vote, {"bvps": bvps, "price": price, "pb": pb, "sector": sector}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def calculate_valuation_score(ticker: str) -> dict:
    """
    Compute S3 (valuation, 0-100) for a ticker via majority vote across eight
    independent valuation models (P/E, EV/EBITDA, P/FCF, PEG, Reverse DCF,
    Graham Number, DDM, P/BV). Models that can't be computed ABSTAIN and are
    excluded from the vote tally.
    """
    data = fetch_stock_data(ticker)
    info = data.get("info") or {}
    cashflow = data.get("cashflow")

    pe_vote, pe_details = _vote_pe(info)
    ev_vote, ev_details = _vote_ev_ebitda(info)
    fcf_vote, fcf_details = _vote_p_fcf(info, cashflow)
    peg_vote, peg_details = _vote_peg(info)
    dcf_vote, dcf_details = _vote_reverse_dcf(info)
    graham_vote, graham_details = _vote_graham(info)
    ddm_vote, ddm_details = _vote_ddm(info)
    pbv_vote, pbv_details = _vote_pbv(info)

    votes = {
        "PE": pe_vote,
        "EV_EBITDA": ev_vote,
        "P_FCF": fcf_vote,
        "PEG": peg_vote,
        "Reverse_DCF": dcf_vote,
        "Graham": graham_vote,
        "DDM": ddm_vote,
        "P_BV": pbv_vote,
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
        # Every valid vote agrees on HOLD -- genuine consensus, not a conflict.
        signal = "HOLD"
    else:
        signal = "MIXED SIGNAL"

    mixed_signal_flag = signal == "MIXED SIGNAL"

    if signal == "BUY":
        s3 = 75 + (buy_count / n) * 25
    elif signal == "SELL":
        s3 = 25 - (sell_count / n) * 25
    else:  # "HOLD" or "MIXED SIGNAL"
        s3 = 50.0

    details = {
        "PE": pe_details,
        "EV_EBITDA": ev_details,
        "P_FCF": fcf_details,
        "PEG": peg_details,
        "Reverse_DCF": dcf_details,
        "Graham": graham_details,
        "DDM": ddm_details,
        "P_BV": pbv_details,
        "insufficient_data": insufficient_data,
    }

    return {
        "S3": round(s3, 2),
        "signal": signal,
        "mixed_signal_flag": mixed_signal_flag,
        "votes": votes,
        "buy_count": buy_count,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "abstain_count": abstain_count,
        "details": details,
    }
