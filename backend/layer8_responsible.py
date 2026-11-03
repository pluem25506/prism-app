"""
Layer 8 (S8): Responsible AI / ESG.

- US stocks: real ESG data from yfinance's Sustainalytics-backed
  `Ticker.sustainability` when available.
- Thai stocks (.BK) and any US ticker without ESG coverage: a governance
  proxy from fundamentals (dividend consistency, leverage, earnings
  consistency), since Sustainalytics doesn't cover most SET-listed stocks.
- Hard fallback: neutral 50 with a clear data_quality flag.
"""

import functools

import numpy as np
import yfinance as yf

from .data_fetcher import normalize_dividend_yield


@functools.lru_cache(maxsize=32)
def compute_layer8(ticker: str) -> dict:
    """
    L8 Responsible AI / ESG Score using yfinance.sustainability
    - US stocks: real ESG data when available
    - Thai stocks (.BK): governance proxy from fundamentals
    - Fallback: neutral 50 with clear data_quality flag
    """
    tk = yf.Ticker(ticker)
    is_thai = ticker.upper().endswith(".BK")

    # ── US stocks: try yfinance sustainability ──────────────────
    if not is_thai:
        try:
            sus = tk.sustainability
            if sus is not None and not sus.empty:
                def get_val(key):
                    if key in sus.index:
                        v = sus.loc[key, "Value"]
                        return float(v) if v is not None else None
                    return None

                total_esg   = get_val("totalEsg")
                env_score   = get_val("environmentScore")
                soc_score   = get_val("socialScore")
                gov_score   = get_val("governanceScore")
                controversy = get_val("highestControversy")  # 0–5, lower=better

                if total_esg is not None:
                    # ESG score from Sustainalytics: lower = better (risk score)
                    # 0–10 Negligible, 10–20 Low, 20–30 Medium, 30–40 High, 40+ Severe
                    if total_esg <= 10:   esg_base = 90
                    elif total_esg <= 20: esg_base = 75
                    elif total_esg <= 30: esg_base = 55
                    elif total_esg <= 40: esg_base = 35
                    else:                 esg_base = 20

                    # Controversy penalty (0=none → 5=severe)
                    controversy_penalty = (controversy or 0) * 6

                    s8 = max(10, min(100, esg_base - controversy_penalty))

                    return {
                        "S8": round(s8, 1),
                        "total_esg":        round(total_esg, 1),
                        "env_score":        round(env_score,  1) if env_score  else None,
                        "social_score":     round(soc_score,  1) if soc_score  else None,
                        "gov_score":        round(gov_score,  1) if gov_score  else None,
                        "controversy":      controversy,
                        "data_quality":     "FULL",
                        "source":           "yfinance/Sustainalytics",
                    }
        except Exception:
            pass

    # ── Thai stocks or fallback: governance proxy ───────────────
    # Use dividend consistency + debt discipline as governance proxy
    try:
        info = tk.info or {}
        hist = tk.history(period="3y")

        scores = []
        details = []

        # 1. Dividend consistency (social/governance signal)
        # normalize_dividend_yield() cross-checks against dividendRate/price
        # when available -- the raw dividendYield field's scale is otherwise
        # ambiguous for low-yield stocks (see data_fetcher.py docstring).
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        div_yield = normalize_dividend_yield(info.get("dividendYield"), info.get("dividendRate"), price) or 0
        if div_yield > 0.05:
            scores.append(75); details.append(f"Dividend yield {div_yield*100:.1f}% (consistent)")
        elif div_yield > 0.02:
            scores.append(60); details.append(f"Dividend yield {div_yield*100:.1f}%")
        elif div_yield > 0:
            scores.append(50); details.append(f"Low dividend yield {div_yield*100:.1f}%")
        else:
            scores.append(40); details.append("No dividend")

        # 2. Debt discipline (governance)
        de_ratio = info.get("debtToEquity") or 0
        if de_ratio < 50:
            scores.append(80); details.append(f"D/E {de_ratio:.0f}% (low leverage)")
        elif de_ratio < 150:
            scores.append(60); details.append(f"D/E {de_ratio:.0f}% (moderate)")
        elif de_ratio < 300:
            scores.append(40); details.append(f"D/E {de_ratio:.0f}% (high)")
        else:
            scores.append(25); details.append(f"D/E {de_ratio:.0f}% (very high)")

        # 3. Earnings consistency (governance proxy)
        if not hist.empty and len(hist) >= 60:
            # Count positive-return years
            annual = hist["Close"].resample("YE").last().pct_change().dropna()
            pos_years = int((annual > 0).sum())
            total_years = len(annual)
            if total_years > 0:
                ratio = pos_years / total_years
                score = 40 + ratio * 50
                scores.append(score)
                details.append(f"Positive years {pos_years}/{total_years}")

        if scores:
            s8 = round(np.mean(scores), 1)
            return {
                "S8":           s8,
                "total_esg":    None,
                "div_yield":    round(div_yield * 100, 2),
                "de_ratio":     round(de_ratio, 1),
                "data_quality": "PROXY",
                "source":       "yfinance fundamentals (ESG proxy)",
                "details":      details,
            }

    except Exception:
        pass

    # ── Hard fallback ────────────────────────────────────────────
    return {
        "S8":           50,
        "data_quality": "STUB",
        "source":       "no data",
        "details":      ["ESG data unavailable"],
    }
