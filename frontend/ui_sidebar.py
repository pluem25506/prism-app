import streamlit as st


def render_sidebar() -> tuple[str, bool]:
    """Render the PRISM sidebar (ticker input + Analyze button). Returns (ticker, clicked)."""
    with st.sidebar:
        st.markdown("## PRISM v2.1")

        ticker = st.text_input(
            "Stock Ticker (SET)",
            value=st.session_state.get("selected_ticker") or "PTT.BK",
            help="Use .BK suffix for SET stocks: PTT.BK, KBANK.BK, AOT.BK",
        )
        clicked = st.button("🔍 Analyze", type="primary", width="stretch")

        if st.session_state.get("run_analysis"):
            if st.button("← Home", width="stretch"):
                st.session_state["run_analysis"] = False
                st.session_state["selected_ticker"] = ""
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 My Portfolio (for L7)")

        if "portfolio" not in st.session_state:
            st.session_state["portfolio"] = []

        with st.expander("Add holdings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                new_ticker = st.text_input("Ticker", key="port_ticker", placeholder="e.g. PTT.BK")
            with col2:
                new_shares = st.number_input("Shares", min_value=0.0, step=100.0, key="port_shares")
            if st.button("➕ Add to Portfolio"):
                if new_ticker:
                    st.session_state["portfolio"].append({
                        "ticker": new_ticker.upper().strip(),
                        "shares": new_shares,
                    })
                    st.success(f"Added {new_ticker.upper()} × {new_shares:,.0f}")

        if st.session_state["portfolio"]:
            st.markdown("**Current holdings:**")
            for i, h in enumerate(st.session_state["portfolio"]):
                hcol1, hcol2, hcol3 = st.columns([3, 2, 1])
                hcol1.write(h["ticker"])
                hcol2.write(f"{h['shares']:,.0f} shares")
                if hcol3.button("✕", key=f"del_{i}"):
                    st.session_state["portfolio"].pop(i)
                    st.rerun()
        else:
            st.caption("No holdings added yet — L7 will use neutral score (50)")

        st.divider()
        st.markdown("**Common SET Tickers**")
        st.caption("PTT.BK · KBANK.BK · AOT.BK · SCB.BK · CPALL.BK")

        st.divider()
        st.markdown("**About PRISM**")
        st.caption("9-layer quantitative model. Math computes every number. AI explains results only.")
        st.caption("v2.1 | Bangkok University | Trinity Hackathon 2026")

    return ticker, clicked
