"""
Compare tab: side-by-side PRISM analysis of two tickers.
"""

import streamlit as st

from backend.prism_score import run_full_analysis
from frontend.ui_score import LAYER_INFO, render_score_gauge


def show_compare_tab():
    st.markdown("### ⚖️ เปรียบเทียบ 2 หุ้น")

    col1, col2 = st.columns(2)
    with col1:
        ticker_a = st.text_input(
            "หุ้นตัวที่ 1", placeholder="เช่น AAPL", key="cmp_a"
        ).strip().upper()
    with col2:
        ticker_b = st.text_input(
            "หุ้นตัวที่ 2", placeholder="เช่น PTT.BK", key="cmp_b"
        ).strip().upper()

    if st.button("🔍 เปรียบเทียบ", width="stretch"):
        if ticker_a and ticker_b:
            with st.spinner("กำลังวิเคราะห์..."):
                result_a = None
                result_b = None
                try:
                    result_a = run_full_analysis(ticker_a)
                except Exception as exc:
                    st.error(f"วิเคราะห์ {ticker_a} ไม่สำเร็จ: {exc}")
                try:
                    result_b = run_full_analysis(ticker_b)
                except Exception as exc:
                    st.error(f"วิเคราะห์ {ticker_b} ไม่สำเร็จ: {exc}")

            if result_a and result_b:
                _render_comparison(ticker_a, result_a, ticker_b, result_b)
        else:
            st.warning("กรุณาใส่ชื่อหุ้นทั้ง 2 ตัว")


def _render_comparison(ticker_a, res_a, ticker_b, res_b):
    """Side-by-side comparison of two PRISM analyses."""

    # Score gauges
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {ticker_a}")
        render_score_gauge(res_a["prism_score"], res_a["signal"])
    with col2:
        st.markdown(f"#### {ticker_b}")
        render_score_gauge(res_b["prism_score"], res_b["signal"])

    st.divider()

    # Layer-by-layer comparison table -- reuse the app's real layer names
    # (ui_score.LAYER_INFO) rather than a separate, inconsistent label set.
    st.markdown("#### เปรียบเทียบแต่ละ Layer")

    header = st.columns([2, 1.5, 1.5])
    header[0].markdown("**Layer**")
    header[1].markdown(f"**{ticker_a}**")
    header[2].markdown(f"**{ticker_b}**")

    for key, (layer_num, name, _weight) in LAYER_INFO.items():
        label = f"{layer_num} {name}"
        val_a = res_a.get(key) or 0
        val_b = res_b.get(key) or 0

        # Highlight winner in green
        color_a = "#00C896" if val_a > val_b else ("#D4A017" if val_a == val_b else "#FF4757")
        color_b = "#00C896" if val_b > val_a else ("#D4A017" if val_a == val_b else "#FF4757")

        row = st.columns([2, 1.5, 1.5])
        row[0].markdown(label)
        row[1].markdown(f"<span style='color:{color_a};font-weight:700'>{val_a:.1f}</span>",
                         unsafe_allow_html=True)
        row[2].markdown(f"<span style='color:{color_b};font-weight:700'>{val_b:.1f}</span>",
                         unsafe_allow_html=True)

    # Winner banner
    st.divider()
    if res_a["prism_score"] > res_b["prism_score"]:
        winner = ticker_a
        score_diff = res_a["prism_score"] - res_b["prism_score"]
    elif res_b["prism_score"] > res_a["prism_score"]:
        winner = ticker_b
        score_diff = res_b["prism_score"] - res_a["prism_score"]
    else:
        winner = None
        score_diff = 0.0

    if winner:
        st.markdown(f"""
        <div style="background:#0F2D1A; border:1px solid #00C896;
                    border-radius:10px; padding:1rem; text-align:center;">
            <span style="font-size:1.2rem; color:#00C896; font-weight:800;">
                🏆 {winner} ชนะ (+{score_diff:.1f} คะแนน)
            </span>
        </div>
        """, unsafe_allow_html=True)
