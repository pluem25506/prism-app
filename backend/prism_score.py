"""
PRISM v2.1 master scorer.

Combines the nine PRISM layers (S1-S9) into a single weighted prism_score.
Any layer that fails or is unavailable is dropped and the remaining weights
renormalized; stub layers are excluded from the confidence calculation since
they aren't real data.
"""

import streamlit as st

from .data_fetcher import fetch_stock_data, validate_ticker
from .layer1_quality import calculate_quality_score as compute_layer1
from .layer2_value import calculate_value_score as compute_layer2
from .layer3_valuation import calculate_valuation_score as compute_layer3
from .layer4_risk import calculate_risk_score as compute_layer4
from .layer5_montecarlo import calculate_montecarlo_score as compute_layer5
from .layer6_momentum import compute_layer6
from .layer7_portfolio import compute_layer7
from .layer8_responsible import compute_layer8
from .layer9_contextual import compute_layer9

WEIGHTS = {
    "S1": 0.20,  # L1 Quality Screen
    "S2": 0.15,  # L2 Value Creation
    "S3": 0.20,  # L3 Valuation
    "S4": 0.15,  # L4 Risk & Leverage
    "S5": 0.10,  # L5 Monte Carlo
    "S6": 0.05,  # L6 Time-Series Momentum
    "S7": 0.05,  # L7 Portfolio Fit
    "S8": 0.05,  # L8 Resp. AI / ESG
    "S9": 0.05,  # L9 Contextual Intel
}

STUBS = {}


def _safe_call(fn, ticker):
    """Run a layer's compute function, degrading to an empty result on any failure."""
    try:
        return fn(ticker)
    except Exception as exc:
        return {"error": str(exc)}


def compute_prism_score(ticker: str) -> dict:
    """Run all nine PRISM layers for a ticker and combine them into a single score."""
    validate_ticker(ticker)

    layer1_result = _safe_call(compute_layer1, ticker)
    layer2_result = _safe_call(compute_layer2, ticker)
    layer3_result = _safe_call(compute_layer3, ticker)
    layer4_result = _safe_call(compute_layer4, ticker)
    layer5_result = _safe_call(compute_layer5, ticker)
    layer6_result = _safe_call(compute_layer6, ticker)

    portfolio = st.session_state.get("portfolio", []) if hasattr(st, "session_state") else []
    if portfolio:
        try:
            layer7_result = compute_layer7(ticker, portfolio)
        except Exception as exc:
            layer7_result = {"error": str(exc)}
    else:
        layer7_result = {"S7": 50, "data_quality": "stub", "details": ["No portfolio set"]}

    layer8_result = _safe_call(compute_layer8, ticker)

    fred_key = st.secrets.get("FRED_API_KEY", "") if hasattr(st, "secrets") else ""
    layer9_result = _safe_call(lambda t: compute_layer9(t, fred_key), ticker)

    computed = {
        "S1": layer1_result.get("S1"),
        "S2": layer2_result.get("S2"),
        "S3": layer3_result.get("S3"),
        "S4": layer4_result.get("S4"),
        "S5": layer5_result.get("S5"),
        "S6": layer6_result.get("S6"),
        "S7": layer7_result.get("S7"),
        "S8": layer8_result.get("S8"),
        "S9": layer9_result.get("S9"),
    }

    sub_scores = {}
    for key in WEIGHTS:
        if key in STUBS:
            sub_scores[key] = STUBS[key]
        else:
            sub_scores[key] = computed.get(key)

    available = {k: v for k, v in sub_scores.items() if v is not None}
    total_weight = sum(WEIGHTS[k] for k in available)

    if total_weight == 0:
        prism_score = 0.0
        confidence = 0.0
    else:
        prism_score = sum(WEIGHTS[k] * v for k, v in available.items()) / total_weight
        # Stubs don't count as "real" data -- only genuinely computed layers
        # contribute to the confidence meter.
        real_layers = {k for k in available if k not in STUBS}
        confidence = sum(WEIGHTS[k] for k in real_layers) * 100

    mixed_flag = layer3_result.get("mixed_signal_flag", False)
    if mixed_flag:
        signal = "MIXED SIGNAL"
    elif prism_score >= 65:
        signal = "BUY"
    elif prism_score >= 45:
        signal = "HOLD"
    else:
        signal = "SELL"

    return {
        "ticker": ticker,
        "prism_score": round(prism_score, 2),
        "signal": signal,
        "confidence": round(confidence, 2),
        "available_layers": len(available),
        "sub_scores": sub_scores,
        "layer_details": {
            "L1": layer1_result,
            "L2": layer2_result,
            "L3": layer3_result,
            "L4": layer4_result,
            "L5": layer5_result,
            "L6": layer6_result,
            "L7": layer7_result,
            "L8": layer8_result,
            "L9": layer9_result,
        },
    }


def run_full_analysis(ticker: str) -> dict:
    """Flat view of compute_prism_score for side-by-side ticker comparison:
    {"prism_score": ..., "signal": ..., "S1": ..., ..., "S9": ...}."""
    result = compute_prism_score(ticker)
    flat = {"prism_score": result["prism_score"], "signal": result["signal"]}
    flat.update(result["sub_scores"])
    return flat
