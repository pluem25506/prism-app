import functools
import time

import pandas as pd
import yfinance as yf

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
