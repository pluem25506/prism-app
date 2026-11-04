"""
Layer 9 -- Contextual Intelligence.
S9 = RS_score * 0.50 + Macro_score * 0.50
RS Ratio: stock vs benchmark (yfinance). Macro: GDP growth, CPI, interest
rate (FRED API).

Works for both the live Analyze page (year=None -> trailing 1y ending today,
latest FRED release) and the Backtest engine (an explicit year -> trailing
1y ending at that fiscal year-end, that year's FRED release).
"""

from .data_fetcher import fetch_price_history_long, get_benchmark_ticker, slice_price_window, fetch_macro_year_range


def _is_thai(ticker: str) -> bool:
    return ticker.upper().endswith(".BK")


def compute_rs_ratio(ticker: str, year: int | None = None) -> dict:
    """Relative Strength ratio vs THD (Thai tickers) or ^GSPC (everything
    else) -- Yahoo's own SET index symbols only ever return a single day of
    history via yfinance, confirmed by direct testing."""
    benchmark = get_benchmark_ticker(ticker)
    try:
        stock_history = fetch_price_history_long(ticker)
        bench_history = fetch_price_history_long(benchmark)
        stock_px = slice_price_window(stock_history, year)["Close"] if not stock_history.empty else stock_history
        bench_px = slice_price_window(bench_history, year)["Close"] if not bench_history.empty else bench_history

        if stock_px.empty or bench_px.empty:
            return {"rs_ratio": None, "rs_score": 50, "error": "no data", "benchmark": benchmark}

        idx = stock_px.index.intersection(bench_px.index)
        if len(idx) < 60:
            return {"rs_ratio": None, "rs_score": 50, "error": "insufficient history", "benchmark": benchmark}

        s = stock_px.loc[idx]
        b = bench_px.loc[idx]

        ret_tk = (s.iloc[-1] / s.iloc[0]) - 1
        ret_bm = (b.iloc[-1] / b.iloc[0]) - 1
        rs_ratio = 1.0 if ret_bm == 0 else (1 + ret_tk) / (1 + ret_bm)

        rs_series = s / b
        mom_3m = float(rs_series.iloc[-1] / rs_series.iloc[-63] - 1) if len(idx) >= 63 else 0.0

        if rs_ratio >= 1.20: base = 85
        elif rs_ratio >= 1.10: base = 75
        elif rs_ratio >= 1.00: base = 60
        elif rs_ratio >= 0.90: base = 45
        elif rs_ratio >= 0.80: base = 30
        else: base = 20

        mom_adj = min(10, max(-10, mom_3m * 50))
        rs_score = max(20, min(90, base + mom_adj))

        return {
            "rs_score": round(rs_score, 1), "rs_ratio": round(float(rs_ratio), 4),
            "ret_tk": round(float(ret_tk), 4), "ret_bm": round(float(ret_bm), 4),
            "mom_3m": round(mom_3m, 4), "benchmark": benchmark,
        }
    except Exception as e:
        return {"rs_ratio": None, "rs_score": 50, "error": str(e), "benchmark": benchmark}


def compute_macro_score(ticker: str, fred_key: str | None = None, year: int | None = None) -> dict:
    import datetime
    resolved_year = year if year is not None else datetime.date.today().year
    is_thai = _is_thai(ticker)
    macro = fetch_macro_year_range(resolved_year - 1, resolved_year, fred_key or "", is_thai=is_thai).get(resolved_year, {})

    gdp_growth = macro.get("gdp_growth")
    cpi_yoy = macro.get("cpi_yoy")
    interest_rate = macro.get("interest_rate")
    rate_flag = "BOT rate approximated" if is_thai else "FRED: FEDFUNDS"

    scores = []
    if gdp_growth is not None:
        if gdp_growth >= 3.0: scores.append((85, 0.35))
        elif gdp_growth >= 1.5: scores.append((65, 0.35))
        elif gdp_growth >= 0: scores.append((45, 0.35))
        else: scores.append((20, 0.35))

    if cpi_yoy is not None:
        if 0 <= cpi_yoy < 3.0: scores.append((80, 0.35))
        elif cpi_yoy < 5.0: scores.append((55, 0.35))
        elif cpi_yoy < 8.0: scores.append((35, 0.35))
        else: scores.append((15, 0.35))

    if interest_rate is not None:
        if interest_rate <= 2.0: scores.append((85, 0.30))
        elif interest_rate <= 3.5: scores.append((70, 0.30))
        elif interest_rate <= 5.0: scores.append((50, 0.30))
        elif interest_rate <= 6.5: scores.append((30, 0.30))
        else: scores.append((15, 0.30))

    if not scores:
        macro_score = 50
        macro_flag = "FRED key not set or data unavailable -- macro defaulted to neutral (50)"
    else:
        total_w = sum(w for _, w in scores)
        macro_score = sum(sc * w for sc, w in scores) / total_w
        macro_flag = rate_flag

    return {
        "gdp_growth": round(gdp_growth, 2) if gdp_growth is not None else None,
        "cpi_yoy": round(cpi_yoy, 2) if cpi_yoy is not None else None,
        "interest_rate": round(interest_rate, 2) if interest_rate is not None else None,
        "macro_score": round(macro_score, 1), "macro_flag": macro_flag, "is_thai": is_thai,
    }


def compute_layer9(ticker: str, fred_key: str | None = None, year: int | None = None) -> dict:
    rs = compute_rs_ratio(ticker, year)
    macro = compute_macro_score(ticker, fred_key, year)

    rs_score = rs.get("rs_score", 50)
    macro_score = macro.get("macro_score", 50)
    S9 = rs_score * 0.50 + macro_score * 0.50

    rs_ok = rs.get("rs_ratio") is not None
    macro_ok = macro.get("gdp_growth") is not None
    quality = "FULL" if (rs_ok and macro_ok) else ("PARTIAL" if (rs_ok or macro_ok) else "STUB")

    details = []
    if rs.get("rs_ratio"):
        d = "outperforming" if rs["rs_ratio"] > 1.05 else ("underperforming" if rs["rs_ratio"] < 0.95 else "in-line with")
        details.append(f"RS Ratio {rs['rs_ratio']:.2f} -- {ticker} is {d} {rs['benchmark']}")
        if rs.get("mom_3m") is not None:
            m = "improving" if rs["mom_3m"] > 0.02 else ("weakening" if rs["mom_3m"] < -0.02 else "stable")
            details.append(f"RS 3M Momentum {rs['mom_3m']*100:+.1f}% -- relative strength is {m}")
    else:
        details.append(rs.get("error", "RS data unavailable"))

    if macro.get("gdp_growth") is not None: details.append(f"GDP Growth (YoY): {macro['gdp_growth']:.1f}%")
    if macro.get("cpi_yoy") is not None: details.append(f"CPI Inflation (YoY): {macro['cpi_yoy']:.1f}%")
    if macro.get("interest_rate") is not None: details.append(f"Interest Rate: {macro['interest_rate']:.2f}%")
    if macro.get("macro_flag"): details.append(f"Note: {macro['macro_flag']}")

    return {"S9": round(S9, 1), "rs": rs, "macro": macro, "details": details, "data_quality": quality}
