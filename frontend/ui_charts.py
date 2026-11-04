import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from backend.data_fetcher import get_currency

DARK_BG = "#0A1628"
DARK_PAPER = "#0A1628"
TEXT_COLOR = "white"


def render_montecarlo_chart(mc_result: dict, ticker: str = ""):
    paths = mc_result["paths_sample"]  # shape (100, 252)
    days = list(range(252))
    current = mc_result["current_price"]
    currency, symbol = get_currency(ticker)

    fig = go.Figure()

    # 100 sample paths (light gray, low opacity)
    for path in paths:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=path,
                mode="lines",
                line=dict(color="gray", width=0.5),
                opacity=0.15,
                showlegend=False,
            )
        )

    # P10 / P50 / P90 highlighted lines
    p10_path = paths[np.argmin(np.abs(paths[:, -1] - mc_result["P10"]))]
    p50_path = paths[np.argmin(np.abs(paths[:, -1] - mc_result["P50"]))]
    p90_path = paths[np.argmin(np.abs(paths[:, -1] - mc_result["P90"]))]

    fig.add_trace(
        go.Scatter(
            x=days,
            y=p10_path,
            name=f'P10: {symbol}{mc_result["P10"]:.2f}',
            line=dict(color="#C0392B", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p50_path,
            name=f'P50: {symbol}{mc_result["P50"]:.2f}',
            line=dict(color="white", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p90_path,
            name=f'P90: {symbol}{mc_result["P90"]:.2f}',
            line=dict(color="#27AE60", width=2),
        )
    )

    # Current price horizontal line
    fig.add_hline(
        y=current,
        line_dash="dash",
        line_color="#D4A017",
        annotation_text=f"Current: {symbol}{current:.2f}",
    )

    fig.update_layout(
        title="Monte Carlo Price Paths (10,000 simulations · Student-t v=5)",
        xaxis_title="Trading Days (1 year)",
        yaxis_title=f"Price ({currency})",
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_PAPER,
        font=dict(color=TEXT_COLOR),
        legend=dict(bgcolor="#1A3A5C"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Prob. of loss: {mc_result['prob_loss']*100:.1f}% · "
        f"Median upside: {mc_result['upside_P50']*100:+.1f}%"
    )


def render_risk_table(risk_result: dict):
    data = {
        "Metric": [
            "VaR 95%",
            "VaR 99%",
            "CVaR 95%",
            "CVaR 99%",
            "Beta",
            "Sharpe Ratio",
            "GARCH Vol (ann.)",
        ],
        "Value": [
            f"{risk_result['VaR_95']*100:.2f}%",
            f"{risk_result['VaR_99']*100:.2f}%",
            f"{risk_result['CVaR_95']*100:.2f}%",
            f"{risk_result['CVaR_99']*100:.2f}%",
            f"{risk_result['beta']:.2f}",
            f"{risk_result['sharpe']:.2f}",
            f"{risk_result['garch_vol_annualized']*100:.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")


def render_leverage_table(l8_result: dict):
    data_quality = l8_result.get("data_quality", "N/A")

    if data_quality == "stub":
        st.caption("Not enough year-over-year fundamentals to compute operating/financial leverage — defaulted to neutral score.")
        return

    data = {
        "Metric": ["DOL (Operating Leverage)", "DFL (Financial Leverage)", "DTL (Total Leverage)", "Data Quality"],
        "Value": [
            f"{l8_result.get('dol', 'N/A')}",
            f"{l8_result.get('dfl', 'N/A')}",
            f"{l8_result.get('dtl', 'N/A')}",
            data_quality,
        ],
    }
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")
    for note in l8_result.get("details", []):
        st.caption(f"• {note}")


def render_contextual_intel(l9_result: dict):
    rs = l9_result.get("rs") or {}
    macro = l9_result.get("macro") or {}
    data_quality = l9_result.get("data_quality", "N/A")

    rs_ratio = rs.get("rs_ratio")
    benchmark = rs.get("benchmark")
    rs_value = f"{rs_ratio:.3f} vs {benchmark}" if rs_ratio is not None and benchmark else "N/A"

    gdp_growth = macro.get("gdp_growth")
    cpi_yoy = macro.get("cpi_yoy")
    interest_rate = macro.get("interest_rate")

    data = {
        "Metric": ["RS Ratio", "GDP Growth (YoY)", "CPI Inflation (YoY)", "Interest Rate", "Data Quality"],
        "Value": [
            rs_value,
            f"{gdp_growth:.1f}%" if gdp_growth is not None else "N/A",
            f"{cpi_yoy:.1f}%" if cpi_yoy is not None else "N/A",
            f"{interest_rate:.2f}%" if interest_rate is not None else "N/A",
            str(data_quality).upper(),
        ],
    }
    st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch")

    for note in l9_result.get("details", []):
        st.caption(f"• {note}")


def render_score_bar(result: dict):
    LAYER_NAMES = [
        "L1 Quality",
        "L2 Value",
        "L3 Valuation",
        "L4 Risk",
        "L5 MonteCarlo",
        "L6 TimeSeries",
        "L7 Portfolio",
        "L8 Resp.AI",
        "L9 Context",
    ]
    scores = [result["sub_scores"].get(f"S{i}", 0) or 0 for i in range(1, 10)]
    colors = ["#27AE60" if s >= 70 else "#D4A017" if s >= 45 else "#C0392B" for s in scores]

    fig = go.Figure(
        go.Bar(
            x=LAYER_NAMES,
            y=scores,
            marker_color=colors,
            text=[f"{s:.0f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Sub-Score Breakdown by Layer",
        yaxis=dict(range=[0, 110]),
        plot_bgcolor=DARK_BG,
        paper_bgcolor=DARK_PAPER,
        font=dict(color=TEXT_COLOR),
    )
    st.plotly_chart(fig, width="stretch")
