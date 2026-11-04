"""
Layer 6 -- Time-Series Momentum.

S6 = weighted average of 6 momentum signals, all from price/volume history
(no API key needed): Price Momentum, 50/200D Moving Average Cross, RSI(14),
MACD, Volume Trend, and Bollinger %B.

Works for both the live Analyze page (year=None -> trailing ~1y ending
today) and the Backtest engine (an explicit year -> trailing ~1y ending at
that fiscal year-end), from the same long price history sliced to the
right window instead of a fresh "last 1y from today" download each time.
"""

from .data_fetcher import fetch_price_history_long, slice_price_window


def _bollinger_score(closes, period: int = 20, n_std: float = 2.0):
    if len(closes) < period:
        return None
    window = closes.iloc[-period:]
    mid = float(window.mean())
    std = float(window.std())
    if std == 0:
        return 55.0
    upper, lower = mid + n_std * std, mid - n_std * std
    percent_b = (float(closes.iloc[-1]) - lower) / (upper - lower)
    if percent_b <= 0.2:
        return 75.0
    if percent_b >= 0.8:
        return 30.0
    return 55.0


def compute_layer6(ticker: str, year: int | None = None, as_of_date=None) -> dict:
    """as_of_date, when given, takes priority over `year` for the window's
    end point -- lets the Backtest engine refresh this purely price-based
    layer more often than once a fiscal year (Daily/Weekly/Monthly
    rebalancing), while fundamentals-based layers stay pinned to `year`."""
    price_history = fetch_price_history_long(ticker)
    as_of = as_of_date if as_of_date is not None else year
    df = slice_price_window(price_history, as_of, lookback_days=400)  # a little slack for the 200D MA warm-up

    if df.empty or len(df) < 60:
        return {"S6": 50, "data_quality": "stub", "details": ["Insufficient price history"]}

    close = df["Close"]
    volume = df["Volume"]

    signals = {}

    # 1. Price Momentum (3M=63d, 6M=126d, 12M=252d) -- weight 0.25
    mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0
    mom_6m = float(close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0
    mom_12m = float(close.iloc[-1] / close.iloc[0] - 1) * 100
    mom_avg = (mom_3m + mom_6m + mom_12m) / 3
    if mom_avg >= 20: mom_score = 90
    elif mom_avg >= 10: mom_score = 75
    elif mom_avg >= 0: mom_score = 55
    elif mom_avg >= -10: mom_score = 35
    else: mom_score = 15
    signals["momentum"] = (mom_score, 0.25)

    # 2. Moving Average (50D vs 200D) -- weight 0.20
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(len(close)).mean().iloc[-1])
    price = float(close.iloc[-1])
    if price > ma50 > ma200: ma_score = 90
    elif price > ma50: ma_score = 70
    elif price > ma200: ma_score = 50
    elif ma50 > ma200: ma_score = 30
    else: ma_score = 15
    signals["ma_cross"] = (ma_score, 0.20)

    # 3. RSI (14-day) -- weight 0.18
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    if rsi >= 70: rsi_score = 30
    elif rsi >= 55: rsi_score = 75
    elif rsi >= 45: rsi_score = 55
    elif rsi >= 30: rsi_score = 35
    else: rsi_score = 70
    signals["rsi"] = (rsi_score, 0.18)

    # 4. MACD Signal -- weight 0.15
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    macd_val = float(macd_line.iloc[-1])
    hist_val = float(histogram.iloc[-1])
    if macd_val > 0 and hist_val > 0: macd_score = 80
    elif macd_val > 0 and hist_val < 0: macd_score = 55
    elif macd_val < 0 and hist_val > 0: macd_score = 50
    else: macd_score = 25
    signals["macd"] = (macd_score, 0.15)

    # 5. Bollinger %B -- weight 0.12
    bollinger_score = _bollinger_score(close)
    if bollinger_score is not None:
        signals["bollinger"] = (bollinger_score, 0.12)

    # 6. Volume Trend -- weight 0.10
    vol_20d = float(volume.rolling(20).mean().iloc[-1])
    vol_60d = float(volume.rolling(60).mean().iloc[-1]) if len(volume) >= 60 else vol_20d
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0
    if vol_ratio >= 1.3: vol_score = 80
    elif vol_ratio >= 1.1: vol_score = 65
    elif vol_ratio >= 0.9: vol_score = 50
    elif vol_ratio >= 0.7: vol_score = 35
    else: vol_score = 20
    signals["volume"] = (vol_score, 0.10)

    total_weight = sum(w for _, w in signals.values())
    S6 = sum(sc * w for sc, w in signals.values()) / total_weight

    details = [
        f"Price Momentum (avg 3/6/12M): {mom_avg:+.1f}%",
        f"MA Cross: Price={price:.2f} vs MA50={ma50:.2f} vs MA200={ma200:.2f}",
        f"RSI(14): {rsi:.1f}",
        f"MACD: {'bullish' if macd_val > 0 else 'bearish'} (histogram {'positive' if hist_val > 0 else 'negative'})",
        f"Volume 20D/60D ratio: {vol_ratio:.2f}x",
    ]

    return {
        "S6": round(S6, 1),
        "rsi": round(rsi, 1),
        "macd_bullish": macd_val > 0,
        "above_ma50": price > ma50,
        "above_ma200": price > ma200,
        "mom_3m": round(mom_3m, 2), "mom_6m": round(mom_6m, 2), "mom_12m": round(mom_12m, 2),
        "vol_ratio": round(vol_ratio, 2),
        "signals": {k: {"score": v[0], "weight": v[1]} for k, v in signals.items()},
        "details": details,
        "data_quality": "full",
        "year": year,
    }
