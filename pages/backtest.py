"""
PRISM Backtest Engine -- standalone page.

# -----------------------------------------------------------------
# SECRETS REQUIRED (add to .streamlit/secrets.toml -- GITIGNORED):
#   FRED_API_KEY = "your_fred_api_key_here"
#   Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
#
# For Streamlit Cloud deployment:
#   Settings -> Secrets -> paste FRED_API_KEY="..."
#   GEMINI_API_KEY is already set for the main app and isn't needed here.
# -----------------------------------------------------------------

Note: emoji were dropped from this page's labels/buttons to match the
rest of the app, which had every emoji removed project-wide.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backend.prism_score import WEIGHTS
from frontend.ui_theme import inject_sidebar_nav_style
from utils.prism_score import LAYER_LABELS, run_backtest

# S7 Portfolio Fit depends on the user's *current* holdings -- it has no
# historical meaning, so backend.prism_score always returns it as None for
# a historical year. Excluded from the backtest's own layer displays (it'd
# just be a permanently-blank row/line otherwise).
BACKTEST_LAYER_KEYS = [k for k in WEIGHTS if k != "S7"]

st.set_page_config(page_title="PRISM Backtest Engine", layout="wide")
inject_sidebar_nav_style()

BG = "#0B1120"
CARD_BG = "#131D30"
GOLD = "#C9A84C"
GOLD_LIGHT = "#E6C670"
GREEN = "#4ADE80"
AMBER = "#FBB040"
RED = "#F87171"
BORDER = "#243350"
BLUE = "#60A5FA"

SIGNAL_COLORS = {"BUY": GREEN, "HOLD": AMBER, "SELL": RED}

PERIOD_OPTIONS = {"10 Years": 10, "15 Years": 15, "20 Years": 20, "Max Available": "max"}
# Monthly/Weekly/Daily recompute the price-based layers (Risk, Time-Series
# Momentum, Contextual Intel) fresh at every rebalance date instead of once
# a year -- the fundamentals-based layers (Quality, Value, Valuation, Monte
# Carlo, Leverage) still only update once a fiscal year regardless, since
# yfinance's annual statements don't change any more often than that.
REBALANCE_OPTIONS = {
    "Annual (Jan 1)": "annual", "Semi-Annual": "semiannual", "Quarterly": "quarterly",
    "Monthly": "monthly", "Weekly": "weekly", "Daily": "daily",
}


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}20;color:{color};border:1px solid {color};'
        f'padding:3px 10px;border-radius:4px;font-weight:700;font-size:13px;">{text}</span>'
    )


st.markdown(
    f"""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <h1 style="margin:0; font-weight:800;">PRISM Backtest Engine <span style="color:{GOLD};">v2.1</span></h1>
        <p style="color:#8899AA; margin-top:0.3rem;">9-layer historical scoring and strategy simulation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Backtest Settings")

# A watchlist card's "BACKTEST" link (see frontend/ui_landing.py) lands here
# as ?ticker=XYZ -- prefill the field with it and auto-run once per distinct
# incoming ticker, same auto-run convenience pattern as the main app's
# landing-page cards (app.py's pending_auto_analyze).
query_ticker = st.query_params.get("ticker")
default_ticker = (query_ticker or "AAPL").upper().strip()

ticker = st.sidebar.text_input("Ticker Symbol", value=default_ticker).upper().strip()
period_label = st.sidebar.selectbox("Lookback Period", list(PERIOD_OPTIONS.keys()), index=0)
rebalance_label = st.sidebar.selectbox("Rebalance Frequency", list(REBALANCE_OPTIONS.keys()), index=0)
if REBALANCE_OPTIONS[rebalance_label] in ("daily", "weekly") and period_label in ("15 Years", "20 Years", "Max Available"):
    st.sidebar.caption(
        f"{rebalance_label} rebalancing over {period_label.lower()} recomputes risk/momentum/contextual "
        "scores at every one of those dates -- this run may take a couple of minutes the first time."
    )

benchmark_choice = st.sidebar.selectbox("Benchmark", ["SPY", "QQQ", "DIA", "BTC-USD", "Custom..."], index=0)
if benchmark_choice == "Custom...":
    benchmark = st.sidebar.text_input("Custom Benchmark Ticker", "MSFT").upper().strip()
else:
    benchmark = benchmark_choice

buy_threshold = st.sidebar.slider("BUY Threshold (>=)", 50, 90, 65)
sell_threshold = st.sidebar.slider("SELL Threshold (<)", 10, 60, 45)
risk_free_when_out = st.sidebar.checkbox("Add risk-free rate when not invested")

run_btn = st.sidebar.button("Run Backtest", type="primary", width="stretch")

try:
    has_fred_key = bool(st.secrets.get("FRED_API_KEY"))
except Exception:
    has_fred_key = False
if not has_fred_key:
    st.sidebar.caption("FRED API key not set -- risk-free rate will default to 4%.")

auto_run = bool(query_ticker) and st.session_state.get("bt_autorun_ticker") != query_ticker
if auto_run:
    st.session_state["bt_autorun_ticker"] = query_ticker

if run_btn or auto_run:
    period_years = PERIOD_OPTIONS[period_label]
    rebalance = REBALANCE_OPTIONS[rebalance_label]
    try:
        with st.spinner("Computing PRISM scores..."):
            result = run_backtest(
                ticker, benchmark, period_years, rebalance,
                buy_threshold, sell_threshold, risk_free_when_out,
            )
        st.session_state["bt_result"] = result
        st.session_state["bt_error"] = None
    except ValueError as exc:
        st.session_state["bt_result"] = None
        st.session_state["bt_error"] = str(exc)
    except Exception as exc:
        st.session_state["bt_result"] = None
        st.session_state["bt_error"] = f"Backtest failed: {exc}"

error = st.session_state.get("bt_error")
result = st.session_state.get("bt_result")

if error:
    st.error(error)
elif result is None:
    st.info("Configure settings in the sidebar and click **Run Backtest**.")
else:
    scores_df = result["scores"]
    stats = result["stats"]
    trade_log = result["trade_log"]
    daily_prices = result["daily_prices"]
    equity = result["equity_curve"]
    benchmark_curve = result["benchmark_curve"]

    if len(scores_df) < 3:
        st.warning(f"Only {len(scores_df)} year(s) of price data available for {result['ticker']}. Results may be limited.")
    if scores_df["data_quality"].eq("minimal").mean() > 0.5:
        st.info(
            f"{result['ticker']} has little or no annual fundamentals data from yfinance for most of this "
            "period (common for crypto, or years older than ~4 fiscal years back). Scores for those years "
            "rely on price momentum and relative strength only."
        )
    if not trade_log:
        st.warning("No BUY/SELL signals triggered across this period. Try adjusting the thresholds in the sidebar.")

    # --- Ticker strip ---
    latest_year = scores_df.index.max()
    latest = scores_df.loc[latest_year]
    strip = st.columns(6)
    with strip[0]:
        st.metric("Ticker", result["ticker"])
    with strip[1]:
        st.metric(f"PRISM Score ({latest_year})", f"{latest['score']:.1f}")
    with strip[2]:
        st.markdown("**Signal**")
        st.markdown(_badge(latest["signal"], SIGNAL_COLORS[latest["signal"]]), unsafe_allow_html=True)
    with strip[3]:
        st.metric("Trades", stats["strategy"]["n_trades"])
    with strip[4]:
        win_rate = stats["strategy"]["win_rate"]
        st.metric("Win Rate", f"{win_rate:.0f}%" if win_rate is not None else "N/A")
    with strip[5]:
        st.metric(f"Alpha vs {result['benchmark']}", f"{stats['alpha']:+.1f}pp")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Cumulative Return", "Candlestick", "Score Timeline"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="PRISM Strategy",
                                  line=dict(color=GOLD, width=2.5)))
        fig.add_trace(go.Scatter(x=benchmark_curve.index, y=benchmark_curve.values, name=f"{result['benchmark']} (Buy & Hold)",
                                  line=dict(color=BLUE, width=2, dash="dash")))

        for t in trade_log:
            if t["date"] not in equity.index:
                continue
            is_buy = t["action"] == "BUY"
            fig.add_trace(go.Scatter(
                x=[t["date"]], y=[equity.loc[t["date"]]], mode="markers",
                marker=dict(symbol="triangle-up" if is_buy else "triangle-down", size=13,
                            color=GREEN if is_buy else RED),
                name=t["action"], showlegend=False,
                hovertemplate=f"{t['action']} @ {t['price']:.2f}<br>Score {t['score']:.1f}<extra></extra>",
            ))

        fig.add_annotation(x=equity.index[-1], y=equity.iloc[-1],
                            text=f"{stats['strategy']['total_return']:+.1f}%",
                            showarrow=True, arrowhead=2, font=dict(color=GOLD, size=13), ax=40)
        fig.add_annotation(x=benchmark_curve.index[-1], y=benchmark_curve.iloc[-1],
                            text=f"{stats['benchmark']['total_return']:+.1f}%",
                            showarrow=True, arrowhead=2, font=dict(color=BLUE, size=13), ax=40, ay=20)

        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="white", height=480,
                           legend=dict(orientation="h", y=1.08), margin=dict(t=40, b=10))
        fig.update_xaxes(gridcolor=BORDER)
        fig.update_yaxes(gridcolor=BORDER, title="Growth of $100")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        show_volume = st.checkbox("Show Volume")
        rows = 3 if show_volume else 2
        row_heights = [0.6, 0.25, 0.15] if show_volume else [0.7, 0.3]

        fig2 = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)

        fig2.add_trace(go.Candlestick(
            x=daily_prices.index, open=daily_prices["Open"], high=daily_prices["High"],
            low=daily_prices["Low"], close=daily_prices["Close"],
            increasing_line_color=GREEN, decreasing_line_color=RED, name="Price",
        ), row=1, col=1)

        ma50 = daily_prices["Close"].rolling(50).mean()
        ma200 = daily_prices["Close"].rolling(200).mean()
        fig2.add_trace(go.Scatter(x=daily_prices.index, y=ma50, name="50D MA",
                                   line=dict(color=GOLD, width=1.2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=daily_prices.index, y=ma200, name="200D MA",
                                   line=dict(color=BLUE, width=1.2)), row=1, col=1)

        for t in trade_log:
            fig2.add_vline(x=t["date"], line_dash="dash",
                            line_color=GREEN if t["action"] == "BUY" else RED, row=1, col=1)

        layer_cols = BACKTEST_LAYER_KEYS
        year_starts, bar_colors, bar_scores, hover_texts = [], [], [], []
        for year, row in scores_df.iterrows():
            year_rows = daily_prices[daily_prices.index.year == year]
            if year_rows.empty:
                continue
            year_starts.append(year_rows.index[0])
            bar_colors.append(SIGNAL_COLORS[row["signal"]])
            bar_scores.append(row["score"])
            layer_lines = "<br>".join(
                f"{k}: {row[k]:.1f}" if pd.notna(row[k]) else f"{k}: N/A" for k in layer_cols
            )
            hover_texts.append(f"<b>{year}</b> ({row['signal']})<br>Score: {row['score']:.1f}<br>{layer_lines}")

        fig2.add_trace(go.Bar(x=year_starts, y=bar_scores, marker_color=bar_colors, name="PRISM Score",
                               hovertext=hover_texts, hoverinfo="text", width=86400000 * 200),
                       row=2, col=1)

        if show_volume:
            fig2.add_trace(go.Bar(x=daily_prices.index, y=daily_prices["Volume"],
                                   marker_color=BORDER, name="Volume"), row=rows, col=1)

        fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="white", height=750,
                            xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.05),
                            margin=dict(t=40, b=10))
        fig2.update_xaxes(
            gridcolor=BORDER,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor=CARD_BG, font=dict(color="white"), activecolor=GOLD,
            ),
            row=1, col=1,
        )
        fig2.update_yaxes(gridcolor=BORDER, range=[0, 100], row=2, col=1)
        fig2.update_yaxes(gridcolor=BORDER)
        st.plotly_chart(fig2, width="stretch")

    with tab3:
        layer_cols = BACKTEST_LAYER_KEYS
        bar_colors = [SIGNAL_COLORS[s] for s in scores_df["signal"]]
        hover_texts = []
        for year, row in scores_df.iterrows():
            layer_lines = "<br>".join(
                f"{k}: {row[k]:.1f}" if pd.notna(row[k]) else f"{k}: N/A" for k in layer_cols
            )
            hover_texts.append(f"<b>{year}</b><br>Score: {row['score']:.1f}<br>{layer_lines}")

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=scores_df.index.astype(str), y=scores_df["score"],
                               marker_color=bar_colors, hovertext=hover_texts, hoverinfo="text"))
        fig3.add_hline(y=buy_threshold, line_dash="dash", line_color=GREEN,
                        annotation_text="BUY threshold", annotation_font_color=GREEN)
        fig3.add_hline(y=sell_threshold, line_dash="dash", line_color=RED,
                        annotation_text="SELL threshold", annotation_font_color=RED)
        fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="white", height=450,
                            margin=dict(t=30, b=10))
        fig3.update_xaxes(gridcolor=BORDER, type="category")
        fig3.update_yaxes(gridcolor=BORDER, range=[0, 100])
        st.plotly_chart(fig3, width="stretch")

        st.markdown("#### Layer Breakdown by Year")
        heat = go.Figure(data=go.Heatmap(
            z=scores_df[layer_cols].T.values,
            x=scores_df.index.astype(str),
            y=[f"{k} {LAYER_LABELS.get(k, '')}" for k in layer_cols],
            colorscale="RdYlGn", zmin=0, zmax=100,
            hovertemplate="Year %{x}<br>%{y}: %{z:.1f}<extra></extra>",
            colorbar=dict(title="Score"),
        ))
        heat.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color="white", height=360,
                            margin=dict(t=20, b=10))
        st.plotly_chart(heat, width="stretch")

    st.divider()
    st.markdown("#### Performance Summary")

    s, b = stats["strategy"], stats["benchmark"]
    win_rate_str = f"{s['win_rate']:.0f}%" if s["win_rate"] is not None else "N/A"
    n_sells = sum(1 for t in trade_log if t["action"] == "SELL")
    n_wins = sum(1 for t in trade_log if t["action"] == "SELL" and t.get("pnl_pct", 0) > 0)

    summary_rows = [
        ("Total Return", f"{s['total_return']:.1f}%", f"{b['total_return']:.1f}%",
         f"{s['total_return'] - b['total_return']:+.1f}pp"),
        ("CAGR", f"{s['cagr']:.1f}%", f"{b['cagr']:.1f}%", f"{s['cagr'] - b['cagr']:+.1f}pp"),
        ("Sharpe Ratio", f"{s['sharpe']:.2f}", f"{b['sharpe']:.2f}", f"{s['sharpe'] - b['sharpe']:+.2f}"),
        ("Max Drawdown", f"{s['max_drawdown']:.1f}%", f"{b['max_drawdown']:.1f}%",
         f"{s['max_drawdown'] - b['max_drawdown']:+.1f}pp"),
        ("Win Rate", win_rate_str, "-", f"{n_wins}/{n_sells} trades" if n_sells else "no closed trades"),
        ("# Trades", str(s["n_trades"]), "1 (Buy & Hold)", "-"),
        ("Alpha", f"{stats['alpha']:+.1f}%", "-", f"vs {result['benchmark']}"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "PRISM", "Benchmark", "Delta"])
    st.dataframe(summary_df, hide_index=True, width="stretch")

    csv = scores_df.to_csv()
    st.download_button(
        "Download Results CSV", csv, f"PRISM_{result['ticker']}_backtest.csv", "text/csv",
    )
