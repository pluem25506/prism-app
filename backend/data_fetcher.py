import functools
import os
import time

import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False

_DEFAULT_RISK_FREE_RATE = 0.04

TICKER_NOT_FOUND_MESSAGE = (
    "Ticker not found. Please use a valid SET ticker (e.g. PTT.BK) or US ticker (e.g. AAPL)."
)

# Yahoo's quoteSummary endpoint (backs .info) occasionally hands back a
# near-empty dict on a transient rate-limit/cookie-negotiation hiccup --
# much more common from shared cloud IP ranges like Streamlit Community
# Cloud's than from a home connection -- even though the request itself
# "succeeds" (no exception). .history()/.download() use a separate, less
# restrictive endpoint and keep working fine during the same hiccup, which
# is why prices/charts are fine while the company-profile fields go blank.
# Retry once before accepting a suspiciously empty result.
_INFO_RETRY_ATTEMPTS = 3
_INFO_RETRY_DELAY_S = 2.0


def _looks_like_real_info(info: dict) -> bool:
    if not info:
        return False
    return any(
        info.get(k) is not None
        for k in ("regularMarketPrice", "currentPrice", "longName", "shortName")
    )


def _fetch_info_with_retry(stock) -> dict:
    last = None
    for attempt in range(_INFO_RETRY_ATTEMPTS):
        try:
            info = stock.info
        except Exception:
            info = None
        if _looks_like_real_info(info):
            return info
        last = info
        if attempt < _INFO_RETRY_ATTEMPTS - 1:
            time.sleep(_INFO_RETRY_DELAY_S)
    return last


class TickerNotFoundError(ValueError):
    """Raised when yfinance returns no price history for a ticker."""


def get_currency(ticker: str) -> tuple:
    """Return (currency_code, symbol) for a ticker.

    SET tickers (".BK" suffix) are priced in THB; everything else (US
    tickers like AAPL, TSLA) is priced in USD.
    """
    if ticker.upper().endswith(".BK"):
        return "THB", "฿"
    return "USD", "$"


@functools.lru_cache(maxsize=32)
def fetch_stock_data(ticker: str) -> dict:
    """Fetch raw yfinance data for a ticker. Any field that fails to fetch is None."""
    stock = yf.Ticker(ticker)
    data = {}

    try:
        data["info"] = _fetch_info_with_retry(stock)
    except Exception:
        data["info"] = None

    try:
        data["history_1y"] = stock.history(period="1y")
    except Exception:
        data["history_1y"] = None

    # If .info came back without even a price (the quoteSummary hiccup
    # described above), backfill from .history(), which has proven more
    # reliable on cloud hosts -- better a real price than a blank "N/A".
    if data["info"] is not None and data["history_1y"] is not None and not data["history_1y"].empty:
        if data["info"].get("currentPrice") is None and data["info"].get("regularMarketPrice") is None:
            last_close = data["history_1y"]["Close"].dropna()
            if not last_close.empty:
                data["info"]["currentPrice"] = float(last_close.iloc[-1])

    try:
        data["history_3y"] = stock.history(period="3y")
    except Exception:
        data["history_3y"] = None

    try:
        data["financials"] = stock.financials
    except Exception:
        data["financials"] = None

    try:
        data["balance_sheet"] = stock.balance_sheet
    except Exception:
        data["balance_sheet"] = None

    try:
        data["cashflow"] = stock.cashflow
    except Exception:
        data["cashflow"] = None

    try:
        data["dividends"] = stock.dividends
    except Exception:
        data["dividends"] = None

    return data


def validate_ticker(ticker: str) -> None:
    """Raise TickerNotFoundError if yfinance has no price history for this ticker.

    Meant to be called before running any analysis, so an invalid/unrecognized
    ticker fails fast with a clear message instead of propagating empty/None
    data silently through every downstream layer.
    """
    history = fetch_stock_data(ticker).get("history_1y")
    if history is None or history.empty:
        raise TickerNotFoundError(TICKER_NOT_FOUND_MESSAGE)


def normalize_dividend_yield(value, dividend_rate=None, price=None):
    """Normalize info["dividendYield"] to a fraction (e.g. 0.025 for 2.5%).

    yfinance has inconsistently returned this field as a fraction in some
    versions and as an already-multiplied percentage (e.g. 6.67 for 6.67%)
    in others. No real stock pays a >100% fractional yield, so any value
    over 1 is unambiguous: already a percentage, so divide by 100.

    A value <=1 is ambiguous on magnitude alone -- e.g. a raw 0.32 could mean
    "0.32 as a fraction" (a 32%-yielding stock) or "0.32 already meaning
    0.32%" (a low-yield stock like AAPL, confirmed by its dividendRate of
    ~$1.08 against a share price in the hundreds). When dividend_rate
    (dollars/share, info["dividendRate"]) and price are both available, use
    them to compute the true yield and pick whichever reading of `value` is
    closer to it, instead of guessing from magnitude alone.
    """
    if value is None:
        return None
    if value > 1:
        return value / 100

    if dividend_rate is not None and price:
        try:
            true_yield = float(dividend_rate) / float(price)
            as_fraction = value
            as_percent = value / 100
            return as_percent if abs(as_percent - true_yield) < abs(as_fraction - true_yield) else as_fraction
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return value


@functools.lru_cache(maxsize=32)
def fetch_price_history_long(ticker: str, period: str = "max") -> pd.DataFrame:
    """Full daily OHLCV history (default: everything yfinance has), used
    wherever a specific historical window needs to be sliced out (L4/L5/L6/L9
    computed 'as of' a past year) rather than always the trailing 1y from
    today that fetch_stock_data()'s history_1y gives.

    Crypto trades in UTC while equities trade in America/New_York --
    comparing/reindexing two series with different tz offsets would barely
    ever align even on the same calendar date, so every series is dropped to
    date-only, tz-naive here.
    """
    df = yf.Ticker(ticker).history(period=period, interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = df.index.tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])


def get_benchmark_ticker(ticker: str) -> str:
    """THD (iShares MSCI Thailand ETF) for Thai (.BK) tickers -- Yahoo's own
    SET index symbols (^SET.BK, ^SETI, ^SET) only ever return a single day
    of history via yfinance, confirmed by direct testing, not a transient
    outage. ^GSPC (S&P 500) for everything else."""
    return "THD" if ticker.upper().endswith(".BK") else "^GSPC"


def slice_price_window(df: pd.DataFrame, as_of, lookback_days: int = 365) -> pd.DataFrame:
    """Trailing `lookback_days` of daily prices ending at:
      - the most recent available trading day, if as_of is None (live mode)
      - Dec 31 of as_of, if as_of is an int (a fiscal year -- the Backtest
        engine's original per-year granularity)
      - as_of itself, if as_of is a date/Timestamp (the Backtest engine's
        finer-than-annual rebalancing -- a specific rebalance date, not
        just a year-end)

    NOT the unsliced full history; every caller (L4/L5/L6/L9) wants
    "trailing ~1y" either way, just anchored at a different point in time.
    """
    if df is None or df.empty:
        return df
    if as_of is None:
        end = df.index[-1]
    elif isinstance(as_of, int):
        end = pd.Timestamp(year=as_of, month=12, day=31)
    else:
        end = pd.Timestamp(as_of)
    window = df[df.index <= end]
    if window.empty:
        return window
    start = window.index[-1] - pd.Timedelta(days=lookback_days)
    return window[window.index >= start]


@functools.lru_cache(maxsize=32)
def fetch_dividends(ticker: str) -> pd.Series:
    try:
        return yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)


def get_price_series(ticker: str) -> pd.Series:
    """Close prices over the last 1 year.

    Drops NaN entries: yfinance sometimes appends a trailing row (e.g. for an
    in-progress or not-yet-settled session) with no Close price, which would
    otherwise silently poison anything using the latest price (prices.iloc[-1]).
    """
    history = fetch_stock_data(ticker).get("history_1y")
    if history is None or history.empty or "Close" not in history.columns:
        return pd.Series(dtype=float)
    return history["Close"].dropna()


def get_return_series(ticker: str) -> pd.Series:
    """Daily percent returns over the last 1 year."""
    prices = get_price_series(ticker)
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.pct_change().dropna()


def _item_row(df, keywords):
    """Fuzzy row lookup returning the full Series across all available
    columns (years), not just the most recent value -- needed for anything
    that looks up a specific historical fiscal year."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    normalized_keywords = [kw.lower().replace(" ", "") for kw in keywords]
    for row_label in df.index:
        row_norm = str(row_label).lower().replace(" ", "")
        for kw in normalized_keywords:
            if kw in row_norm or row_norm in kw:
                return df.loc[row_label]
    return None


@functools.lru_cache(maxsize=32)
def fetch_fundamentals_annual(ticker: str) -> dict:
    """Annual fundamentals keyed by fiscal year (int), shared by both the
    live Analyze scorer (which uses the latest available year) and the
    Backtest engine (which uses each historical year in turn).

    yfinance's free annual financials/balance_sheet/cashflow only go back
    ~4 fiscal years -- years outside that window simply aren't keys in the
    returned dict. Callers degrade gracefully (skip layers needing them)
    rather than crashing or guessing.
    """
    tk = yf.Ticker(ticker)

    def safe(attr):
        try:
            return getattr(tk, attr)
        except Exception:
            return None

    financials = safe("financials")
    balance_sheet = safe("balance_sheet")
    cashflow = safe("cashflow")
    info = safe("info") or {}

    rows = {
        "revenue": _item_row(financials, ["Total Revenue", "TotalRevenue", "Operating Revenue"]),
        "net_income": _item_row(financials, ["Net Income Common Stockholders", "Net Income", "NetIncome"]),
        "ebitda": _item_row(financials, ["EBITDA", "Normalized EBITDA"]),
        "ebit": _item_row(financials, ["EBIT", "Operating Income"]),
        "interest_expense": _item_row(financials, ["Interest Expense"]),
        "cost_of_revenue": _item_row(financials, ["Cost Of Revenue", "Cost Of Goods Sold", "Reconciled Cost Of Revenue"]),
        "pretax_income": _item_row(financials, ["Pretax Income", "Income Before Tax"]),
        "tax_expense": _item_row(financials, ["Tax Provision", "Income Tax Expense"]),
        "total_assets": _item_row(balance_sheet, ["Total Assets"]),
        "total_equity": _item_row(balance_sheet, ["Total Equity Gross Minority Interest", "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"]),
        "total_debt": _item_row(balance_sheet, ["Total Debt"]),
        "short_term_debt": _item_row(balance_sheet, ["Current Debt", "Short Term Debt"]),
        "long_term_debt": _item_row(balance_sheet, ["Long Term Debt"]),
        "cash": _item_row(balance_sheet, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
        "current_assets": _item_row(balance_sheet, ["Total Current Assets", "Current Assets"]),
        "current_liabilities": _item_row(balance_sheet, ["Total Current Liabilities", "Current Liabilities"]),
        "inventory": _item_row(balance_sheet, ["Inventory"]),
        "retained_earnings": _item_row(balance_sheet, ["Retained Earnings"]),
        "total_liabilities": _item_row(balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"]),
        "accounts_receivable": _item_row(balance_sheet, ["Net Receivables", "Accounts Receivable", "Receivables"]),
        "net_ppe": _item_row(balance_sheet, ["Net PPE", "Property Plant Equipment Net", "Properties Plant And Equipment Net"]),
        "operating_cashflow": _item_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"]),
        "capex": _item_row(cashflow, ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE", "Purchases Of Property Plant And Equipment"]),
        "depreciation": _item_row(cashflow, ["Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation"]),
    }

    years = set()
    for row in rows.values():
        if row is not None:
            years.update(c.year for c in row.index)

    shares_out = info.get("sharesOutstanding")
    sector = info.get("sector", "") or ""

    data = {}
    for year in years:
        def val(row, _year=year):
            if row is None:
                return None
            for c in row.index:
                if c.year == _year:
                    v = row[c]
                    return float(v) if pd.notna(v) else None
            return None

        data[year] = {key: val(row) for key, row in rows.items()}
        data[year]["shares_outstanding"] = shares_out
        data[year]["sector"] = sector

    return data


@functools.lru_cache(maxsize=4)
def _fetch_fred_raw_series(fred_key: str):
    """The actual network call to FRED, cached by API key alone (not by
    year range) -- fetch_macro_year_range() used to be cached per exact
    (start_year, end_year) pair, but every per-year call in a Backtest loop
    passes a different narrow range, so every single year was a fresh
    network round-trip (3-5 FRED series) instead of a cache hit. Fetching
    every series once here and slicing years out of it in-memory in
    fetch_macro_year_range() turns an O(years) number of API calls into O(1).
    Returns None if FRED is unavailable or the key is missing/invalid."""
    if not fred_key or not _FRED_AVAILABLE:
        return None
    try:
        fred = Fred(api_key=fred_key)
        gdp_us = fred.get_series_latest_release("GDPC1").dropna()
        cpi_us = fred.get_series_latest_release("CPIAUCSL").dropna()
        fedfunds = fred.get_series_latest_release("FEDFUNDS").dropna()
        dgs10 = fred.get_series("DGS10").dropna()
        try:
            gdp_th = fred.get_series_latest_release("NYGDPPCAPKDTHA").dropna()
        except Exception:
            gdp_th = None
        try:
            cpi_th = fred.get_series_latest_release("THAICPIALLMINMEI").dropna()
        except Exception:
            cpi_th = None
        return {"gdp_us": gdp_us, "cpi_us": cpi_us, "fedfunds": fedfunds, "dgs10": dgs10, "gdp_th": gdp_th, "cpi_th": cpi_th}
    except Exception:
        return None


def fetch_macro_year_range(start_year: int, end_year: int, fred_key: str = "", is_thai: bool = False) -> dict:
    """{year: {"gdp_growth": ..%, "cpi_yoy": ..%, "interest_rate": ..%,
    "risk_free_rate": fraction}}, one value per year (annual mean/YoY).
    Falls back to all-None metrics + risk_free_rate=4% for every year if
    FRED is unavailable -- never raises.

    is_thai=True tries Thailand-specific GDP/CPI series first (falling back
    to the US series if those come back empty, same as the original
    Thai-aware logic), and approximates the Thai policy rate at a flat 2.5%
    (BOT rate) instead of reading FEDFUNDS -- same approximation the live
    app always used for Thai tickers.
    """
    fallback = {y: {"gdp_growth": None, "cpi_yoy": None, "interest_rate": None,
                     "risk_free_rate": _DEFAULT_RISK_FREE_RATE} for y in range(start_year, end_year + 1)}

    key = fred_key or os.environ.get("FRED_API_KEY", "")
    raw = _fetch_fred_raw_series(key)
    if raw is None:
        return fallback

    gdp_us, cpi_us, fedfunds, dgs10 = raw["gdp_us"], raw["cpi_us"], raw["fedfunds"], raw["dgs10"]
    gdp_th = raw["gdp_th"] if is_thai else None
    cpi_th = raw["cpi_th"] if is_thai else None

    def yoy_for_year(series, year):
        if series is None:
            return None
        try:
            this_year = series[series.index.year == year]
            prior_year = series[series.index.year == year - 1]
            if this_year.empty or prior_year.empty:
                return None
            latest, year_ago = float(this_year.iloc[-1]), float(prior_year.iloc[-1])
            return (latest - year_ago) / abs(year_ago) * 100 if year_ago != 0 else None
        except Exception:
            return None

    def level_for_year(series, year):
        try:
            this_year = series[series.index.year == year]
            return float(this_year.mean()) if not this_year.empty else None
        except Exception:
            return None

    result = {}
    for year in range(start_year, end_year + 1):
        rf_level = level_for_year(dgs10, year)
        gdp_growth = yoy_for_year(gdp_th, year) if is_thai else None
        if gdp_growth is None:
            gdp_growth = yoy_for_year(gdp_us, year)
        cpi_yoy = yoy_for_year(cpi_th, year) if is_thai else None
        if cpi_yoy is None:
            cpi_yoy = yoy_for_year(cpi_us, year)
        interest_rate = 2.5 if is_thai else level_for_year(fedfunds, year)

        result[year] = {
            "gdp_growth": gdp_growth, "cpi_yoy": cpi_yoy, "interest_rate": interest_rate,
            "risk_free_rate": (rf_level / 100) if rf_level is not None else _DEFAULT_RISK_FREE_RATE,
        }
    return result


def _linear_score(value, good, bad, higher_is_better=True):
    """Map `value` onto 0-100 between the `bad` and `good` anchors. Shared by
    every layer that needs a smooth (not bucketed) score from a ratio."""
    if value is None:
        return None
    if good == bad:
        return 100.0 if (value >= good) == higher_is_better else 0.0
    frac = (value - bad) / (good - bad)
    return float(max(0.0, min(1.0, frac)) * 100)


def weighted_average(pairs):
    """pairs: [(score_or_None, weight), ...]. Missing scores are dropped and
    the remaining weights renormalized, so a layer with some unavailable
    sub-metrics still scores fairly on whatever it does have."""
    present = [(score, weight) for score, weight in pairs if score is not None]
    if not present:
        return None
    total_weight = sum(weight for _, weight in present)
    if total_weight == 0:
        return None
    return float(sum(score * weight for score, weight in present) / total_weight)


def get_financial_item(df: pd.DataFrame, keywords: list) -> float:
    """
    Search a financials/balance_sheet DataFrame's row labels for any of the given
    keywords (case-insensitive, whitespace-insensitive) and return the most recent
    (first column) value for the first matching row, or None if nothing matches.

    yfinance row names vary by stock and region, so callers should pass every
    plausible label, e.g. get_financial_item(bs, ["Total Assets", "TotalAssets"]).
    """
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
                try:
                    return float(values.iloc[0])
                except (TypeError, ValueError):
                    continue

    return None
