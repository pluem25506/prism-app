"""
Layer 9 — Contextual Intelligence
S9 = RS_score * 0.50 + Macro_score * 0.50
RS Ratio: stock vs benchmark (yfinance)
Macro: GDP growth, CPI, interest rate (FRED API)
"""

import os
import functools
import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


def _is_thai(ticker: str) -> bool:
    return ticker.upper().endswith(".BK")

def _benchmark(ticker: str) -> str:
    # Yahoo Finance's SET index symbols (^SET.BK, ^SETI, ^SET) only ever
    # return a single day of history via yfinance -- confirmed by direct
    # testing, not a transient outage -- so RS vs ^SET.BK can never compute
    # for any Thai ticker. THD (iShares MSCI Thailand ETF) is a liquid,
    # full-history proxy for the Thai market and is used instead.
    return "THD" if _is_thai(ticker) else "^GSPC"

@functools.lru_cache(maxsize=32)
def _price_series(ticker: str, period: str = "1y") -> pd.Series:
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if data.empty:
        return pd.Series(dtype=float)
    close = data["Close"]
    # .squeeze() alone collapses to a bare scalar (not a Series) when there's
    # only a single row -- e.g. ^SET.BK, which yfinance only ever returns one
    # day of history for. Only squeeze the column axis, never the row axis.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def compute_rs_ratio(ticker: str) -> dict:
    """
    Compute Relative Strength ratio vs an appropriate benchmark.
    Thai stocks (.BK) -> benchmark = THD (see _benchmark() for why not
    ^SET.BK). US stocks -> benchmark = ^GSPC, unchanged.
    """
    benchmark = _benchmark(ticker)
    try:
        stock_px = _price_series(ticker, "1y")
        bench_px = _price_series(benchmark, "1y")

        if stock_px.empty or bench_px.empty:
            return {"rs_ratio": None, "rs_score": 50, "error": "no data", "benchmark": benchmark}

        # Align on common dates
        idx = stock_px.index.intersection(bench_px.index)

        # Lower threshold: need at least 60 trading days
        if len(idx) < 60:
            return {"rs_ratio": None, "rs_score": 50, "error": "insufficient history", "benchmark": benchmark}

        s = stock_px.loc[idx]
        b = bench_px.loc[idx]

        # RS Ratio = 1Y return of ticker / 1Y return of benchmark
        ret_tk = (s.iloc[-1] / s.iloc[0]) - 1
        ret_bm = (b.iloc[-1] / b.iloc[0]) - 1
        rs_ratio = 1.0 if ret_bm == 0 else (1 + ret_tk) / (1 + ret_bm)

        # 3M momentum of the RS ratio itself
        rs_series = s / b
        mom_3m = float(rs_series.iloc[-1] / rs_series.iloc[-63] - 1) if len(idx) >= 63 else 0.0

        # Score: rs_ratio > 1 is outperforming
        if   rs_ratio >= 1.20: base = 85
        elif rs_ratio >= 1.10: base = 75
        elif rs_ratio >= 1.00: base = 60
        elif rs_ratio >= 0.90: base = 45
        elif rs_ratio >= 0.80: base = 30
        else:                  base = 20

        # Momentum adjustment +/-10
        mom_adj = min(10, max(-10, mom_3m * 50))
        rs_score = max(20, min(90, base + mom_adj))

        return {
            "rs_score":  round(rs_score, 1),
            "rs_ratio":  round(float(rs_ratio), 4),
            "ret_tk":    round(float(ret_tk), 4),
            "ret_bm":    round(float(ret_bm), 4),
            "mom_3m":    round(mom_3m, 4),
            "benchmark": benchmark,
        }
    except Exception as e:
        return {"rs_ratio": None, "rs_score": 50, "error": str(e), "benchmark": benchmark}


def compute_macro_score(ticker: str, fred_key: str | None = None) -> dict:
    if not fred_key:
        fred_key = os.environ.get("FRED_API_KEY", "")

    if not fred_key or not FRED_AVAILABLE:
        return {"macro_score": 50, "macro_flag": "FRED key not set — macro defaulted to neutral (50)",
                "gdp_growth": None, "cpi_yoy": None, "interest_rate": None}

    try:
        fred = Fred(api_key=fred_key)
    except Exception as e:
        return {"macro_score": 50, "macro_flag": f"FRED error: {e}",
                "gdp_growth": None, "cpi_yoy": None, "interest_rate": None}

    def yoy(series_id):
        try:
            s = fred.get_series_latest_release(series_id).dropna()
            if len(s) < 13: return None
            latest, year_ago = float(s.iloc[-1]), float(s.iloc[-13])
            return (latest - year_ago) / abs(year_ago) * 100 if year_ago != 0 else None
        except Exception:
            return None

    def latest(series_id):
        try:
            s = fred.get_series_latest_release(series_id).dropna()
            return float(s.iloc[-1]) if not s.empty else None
        except Exception:
            return None

    is_thai = _is_thai(ticker)

    gdp_growth    = yoy("NYGDPPCAPKDTHA") if is_thai else yoy("GDPC1")
    if gdp_growth is None and is_thai:
        gdp_growth = yoy("GDPC1")

    cpi_yoy       = yoy("THAICPIALLMINMEI") if is_thai else yoy("CPIAUCSL")
    if cpi_yoy is None and is_thai:
        cpi_yoy = yoy("CPIAUCSL")

    interest_rate = 2.5 if is_thai else latest("FEDFUNDS")
    rate_flag     = "BOT rate approximated" if is_thai else "FRED: FEDFUNDS"

    scores = []

    if gdp_growth is not None:
        if   gdp_growth >= 3.0: scores.append((85, 0.35))
        elif gdp_growth >= 1.5: scores.append((65, 0.35))
        elif gdp_growth >= 0:   scores.append((45, 0.35))
        else:                   scores.append((20, 0.35))

    if cpi_yoy is not None:
        if   0 <= cpi_yoy < 3.0: scores.append((80, 0.35))
        elif cpi_yoy < 5.0:      scores.append((55, 0.35))
        elif cpi_yoy < 8.0:      scores.append((35, 0.35))
        elif cpi_yoy >= 8.0:     scores.append((15, 0.35))
        else:                    scores.append((40, 0.35))

    if interest_rate is not None:
        if   interest_rate <= 2.0: scores.append((85, 0.30))
        elif interest_rate <= 3.5: scores.append((70, 0.30))
        elif interest_rate <= 5.0: scores.append((50, 0.30))
        elif interest_rate <= 6.5: scores.append((30, 0.30))
        else:                      scores.append((15, 0.30))

    if not scores:
        macro_score = 50
    else:
        total_w = sum(w for _, w in scores)
        macro_score = sum(sc * w for sc, w in scores) / total_w

    return {
        "gdp_growth":    round(gdp_growth, 2)    if gdp_growth    is not None else None,
        "cpi_yoy":       round(cpi_yoy, 2)       if cpi_yoy       is not None else None,
        "interest_rate": round(float(interest_rate), 2) if interest_rate is not None else None,
        "macro_score":   round(macro_score, 1),
        "macro_flag":    rate_flag,
        "is_thai":       is_thai,
    }


def compute_layer9(ticker: str, fred_key: str | None = None) -> dict:
    rs    = compute_rs_ratio(ticker)
    macro = compute_macro_score(ticker, fred_key)

    rs_score    = rs.get("rs_score", 50)
    macro_score = macro.get("macro_score", 50)
    S9 = rs_score * 0.50 + macro_score * 0.50

    rs_ok    = rs.get("rs_ratio") is not None
    macro_ok = macro.get("gdp_growth") is not None

    if rs_ok and macro_ok:   quality = "FULL"
    elif rs_ok or macro_ok:  quality = "PARTIAL"
    else:                    quality = "STUB"

    details = []
    if rs.get("rs_ratio"):
        d = "outperforming" if rs["rs_ratio"] > 1.05 else ("underperforming" if rs["rs_ratio"] < 0.95 else "in-line with")
        details.append(f"RS Ratio {rs['rs_ratio']:.2f} — {ticker} is {d} {rs['benchmark']}")
        if rs.get("mom_3m") is not None:
            m = "improving" if rs["mom_3m"] > 0.02 else ("weakening" if rs["mom_3m"] < -0.02 else "stable")
            details.append(f"RS 3M Momentum {rs['mom_3m']*100:+.1f}% — relative strength is {m}")
    else:
        details.append(rs.get("error", "RS data unavailable"))

    if macro.get("gdp_growth")    is not None: details.append(f"GDP Growth (YoY): {macro['gdp_growth']:.1f}%")
    if macro.get("cpi_yoy")       is not None: details.append(f"CPI Inflation (YoY): {macro['cpi_yoy']:.1f}%")
    if macro.get("interest_rate") is not None: details.append(f"Interest Rate: {macro['interest_rate']:.2f}%")
    if macro.get("macro_flag"):                details.append(f"Note: {macro['macro_flag']}")

    return {
        "S9":           round(S9, 1),
        "rs":           rs,
        "macro":        macro,
        "details":      details,
        "data_quality": quality,
    }
