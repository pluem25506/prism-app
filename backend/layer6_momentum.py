"""
Layer 6 — Time-Series Momentum
S6 = weighted average of 5 momentum signals, all from yfinance (no API key needed)
"""

import functools
import numpy as np
import pandas as pd
import yfinance as yf

@functools.lru_cache(maxsize=32)
def _ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    return df

def compute_layer6(ticker: str) -> dict:
    df = _ohlcv(ticker, "1y")
    if df.empty or len(df) < 60:
        return {"S6": 50, "data_quality": "stub", "details": ["Insufficient price history"]}

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    signals = {}

    # 1. Price Momentum (3M=63d, 6M=126d, 12M=252d) — weight 0.30
    mom_3m  = float(close.iloc[-1] / close.iloc[-63]  - 1) * 100 if len(close) >= 63  else 0
    mom_6m  = float(close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) >= 126 else 0
    mom_12m = float(close.iloc[-1] / close.iloc[0]    - 1) * 100
    mom_avg = (mom_3m + mom_6m + mom_12m) / 3
    if   mom_avg >= 20:  mom_score = 90
    elif mom_avg >= 10:  mom_score = 75
    elif mom_avg >= 0:   mom_score = 55
    elif mom_avg >= -10: mom_score = 35
    else:                mom_score = 15
    signals["momentum"] = (mom_score, 0.30)

    # 2. Moving Average (50D vs 200D) — weight 0.25
    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(len(close)).mean().iloc[-1])
    price = float(close.iloc[-1])
    if price > ma50 > ma200:   ma_score = 90   # golden cross, above both
    elif price > ma50:         ma_score = 70   # above 50D
    elif price > ma200:        ma_score = 50   # above 200D only
    elif ma50 > ma200:         ma_score = 30   # death cross approaching
    else:                      ma_score = 15   # below both, death cross
    signals["ma_cross"] = (ma_score, 0.25)

    # 3. RSI (14-day) — weight 0.20
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])
    if   rsi >= 70:  rsi_score = 30    # overbought = risk
    elif rsi >= 55:  rsi_score = 75    # bullish momentum
    elif rsi >= 45:  rsi_score = 55    # neutral
    elif rsi >= 30:  rsi_score = 35    # weak
    else:            rsi_score = 70    # oversold = potential bounce (contrarian)
    signals["rsi"] = (rsi_score, 0.20)

    # 4. MACD Signal — weight 0.15
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram   = macd_line - signal_line
    macd_val    = float(macd_line.iloc[-1])
    hist_val    = float(histogram.iloc[-1])
    if macd_val > 0 and hist_val > 0:   macd_score = 80
    elif macd_val > 0 and hist_val < 0: macd_score = 55
    elif macd_val < 0 and hist_val > 0: macd_score = 50   # bullish crossover forming
    else:                                macd_score = 25
    signals["macd"] = (macd_score, 0.15)

    # 5. Volume Trend — weight 0.10
    vol_20d  = float(volume.rolling(20).mean().iloc[-1])
    vol_60d  = float(volume.rolling(60).mean().iloc[-1]) if len(volume) >= 60 else vol_20d
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1.0
    if   vol_ratio >= 1.3:  vol_score = 80
    elif vol_ratio >= 1.1:  vol_score = 65
    elif vol_ratio >= 0.9:  vol_score = 50
    elif vol_ratio >= 0.7:  vol_score = 35
    else:                   vol_score = 20
    signals["volume"] = (vol_score, 0.10)

    # Weighted S6
    S6 = sum(sc * w for sc, w in signals.values())

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
        "mom_3m": round(mom_3m, 2),
        "mom_6m": round(mom_6m, 2),
        "mom_12m": round(mom_12m, 2),
        "vol_ratio": round(vol_ratio, 2),
        "signals": {k: {"score": v[0], "weight": v[1]} for k, v in signals.items()},
        "details": details,
        "data_quality": "full",
    }
