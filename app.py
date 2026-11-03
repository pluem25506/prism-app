import streamlit as st

from frontend.ui_sidebar import render_sidebar
from frontend.ui_company import render_company_profile
from frontend.ui_score import render_score_card
from frontend.ui_charts import render_contextual_intel, render_esg_table, render_montecarlo_chart, render_risk_table, render_score_bar
from frontend.ui_ai_summary import render_ai_summary
from frontend.ui_landing import show_landing_page
from frontend.ui_compare import show_compare_tab
from backend.data_fetcher import TickerNotFoundError, fetch_stock_data
from backend.prism_score import compute_prism_score

st.set_page_config(
    page_title="PRISM v2.1 | Quantitative Investment Analysis",
    page_icon="📊",
    layout="wide",
)

# Session state: landing page is shown until the user runs an analysis
# (either via the sidebar or a watchlist card on the landing page).
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = ""
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

ticker, clicked = render_sidebar()

# Set by a landing-page card/search click (see frontend/ui_landing.py) to
# auto-run the analysis for the pre-filled ticker without an extra click.
auto_run = st.session_state.pop("pending_auto_analyze", False)

if clicked or auto_run:
    st.session_state["run_analysis"] = True
    st.session_state["selected_ticker"] = ticker
    with st.spinner("Running 9-layer quantitative analysis..."):
        try:
            st.session_state["result"] = compute_prism_score(ticker)
            st.session_state["ticker"] = ticker
            st.session_state["info"] = fetch_stock_data(ticker)["info"]
        except TickerNotFoundError as exc:
            st.session_state["result"] = None
            st.error(str(exc))
        except Exception as exc:
            st.session_state["result"] = None
            st.error(f"Failed to analyze {ticker}: {exc}")

tab_main, tab_compare = st.tabs(["📊 วิเคราะห์", "⚖️ เปรียบเทียบ"])

with tab_main:
    if not st.session_state.get("run_analysis", False):
        show_landing_page()
    else:
        st.markdown(
            """
            <div style="display:flex; align-items:center;
                        border-bottom:1px solid #1E3A5F;
                        padding-bottom:12px; margin-bottom:20px;">
              <span style="font-size:22px; font-weight:700;
                           color:#D4A017; letter-spacing:2px">PRISM</span>
              <span style="font-size:12px; color:#8899AA;
                           margin-left:8px; margin-top:4px">v2.1 &middot; Predictive Risk &amp; Insight Signal Model</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        result = st.session_state.get("result")

        if result is None:
            st.info("Enter a ticker in the sidebar and click **Analyze** to run the PRISM model.")
        else:
            render_company_profile(st.session_state.get("ticker", ticker), st.session_state.get("info"))

            left_col, right_col = st.columns([6, 4])

            with left_col:
                render_score_card(result)
                render_score_bar(result)

            with right_col:
                render_montecarlo_chart(result["layer_details"]["L5"], result["ticker"])
                render_risk_table(result["layer_details"]["L4"])
                st.markdown("**Responsible AI / ESG (L8)**")
                render_esg_table(result["layer_details"]["L8"])
                st.markdown("**Contextual Intelligence (L9)**")
                render_contextual_intel(result["layer_details"]["L9"])

            render_ai_summary(result)

with tab_compare:
    show_compare_tab()

st.divider()
st.caption(
    "⚠️ PRISM is a quantitative decision-support tool only. "
    "Not investment advice. All models carry assumptions — "
    "verify before acting. For educational purposes only."
)
