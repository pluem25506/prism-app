"""
Landing / home page: "Market Pulse" -- real-time watchlist cards shown before
the user has analyzed any ticker. Clicking a card's Analyze button (or the
quick search bar) pre-fills the sidebar ticker and auto-runs the full
analysis (see app.py's landing/auto-run wiring).
"""

import streamlit as st
import yfinance as yf

WATCHLIST = {
    "🇹🇭 Thai Stocks": [
        {"ticker": "PTT.BK", "name": "PTT"},
        {"ticker": "AOT.BK", "name": "Airports of Thailand"},
        {"ticker": "ADVANC.BK", "name": "Advanced Info Service"},
        {"ticker": "CPALL.BK", "name": "CP ALL"},
        {"ticker": "SCB.BK", "name": "SCB"},
        {"ticker": "KBANK.BK", "name": "KASIKORNBANK"},
        {"ticker": "DELTA.BK", "name": "Delta Electronics"},
        {"ticker": "BBL.BK", "name": "Bangkok Bank"},
    ],
    "🇺🇸 US Stocks": [
        {"ticker": "AAPL", "name": "Apple"},
        {"ticker": "MSFT", "name": "Microsoft"},
        {"ticker": "NVDA", "name": "NVIDIA"},
        {"ticker": "TSLA", "name": "Tesla"},
        {"ticker": "AMZN", "name": "Amazon"},
        {"ticker": "GOOGL", "name": "Alphabet"},
        {"ticker": "META", "name": "Meta"},
        {"ticker": "BRK-B", "name": "Berkshire"},
    ],
}


@st.cache_data(ttl=300)  # cache 5 minutes
def fetch_card_data(ticker: str) -> dict:
    """Fetch price, change%, and 30-day history for a watchlist card."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="30d")
        if hist.empty:
            return None
        # Drop a trailing NaN row (e.g. an in-progress/unsettled session) --
        # otherwise the latest price/sparkline point can come back NaN.
        closes_series = hist["Close"].dropna()
        if closes_series.empty:
            return None
        current = float(closes_series.iloc[-1])
        prev = float(closes_series.iloc[-2]) if len(closes_series) > 1 else current
        change_pct = (current - prev) / prev * 100
        closes = closes_series.tolist()
        currency = "฿" if ticker.upper().endswith(".BK") else "$"
        return {
            "price": current,
            "change_pct": change_pct,
            "closes": closes,
            "currency": currency,
        }
    except Exception:
        return None


def _render_pulse_card(ticker: str, name: str, data):
    """Render a single Market Pulse card."""
    if data is None:
        st.error(f"❌ {ticker}")
        return

    price = data["price"]
    change_pct = data["change_pct"]
    currency = data["currency"]
    closes = data["closes"]

    color = "#00C896" if change_pct >= 0 else "#FF4757"
    arrow = "▲" if change_pct >= 0 else "▼"
    sign = "+" if change_pct >= 0 else ""

    # Sparkline as tiny SVG (30 points -> W=120 H=32)
    if len(closes) >= 2:
        mn, mx = min(closes), max(closes)
        rng = mx - mn if mx != mn else 1
        W, H = 120, 32
        pts = " ".join(
            f"{int(j/(len(closes)-1)*W)},{int(H - (v-mn)/rng*H)}"
            for j, v in enumerate(closes)
        )
        # No leading whitespace/newlines here -- Streamlit's markdown parser
        # treats indented lines as a code block, which renders raw HTML as
        # literal text instead of an actual <svg>.
        sparkline_svg = (
            f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f"</svg>"
        )
    else:
        sparkline_svg = ""

    st.markdown(f"""
    <div style="
        background: #0F1F3D;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        min-height: 110px;
    ">
        <div style="font-size:0.7rem; color:#888; font-weight:600;
                    letter-spacing:0.05em;">{ticker}</div>
        <div style="font-size:0.85rem; font-weight:700;
                    white-space:nowrap; overflow:hidden;
                    text-overflow:ellipsis;">{name}</div>
        <div style="font-size:1.2rem; font-weight:800; margin-top:4px;">
            {currency}{price:,.2f}
        </div>
        <div style="font-size:0.85rem; color:{color}; font-weight:600;">
            {arrow} {sign}{change_pct:.2f}%
        </div>
        {sparkline_svg}
    </div>
    """, unsafe_allow_html=True)

    # Analyze button below card
    if st.button("Analyze", key=f"btn_{ticker}", width="stretch"):
        st.session_state["selected_ticker"] = ticker
        st.session_state["run_analysis"] = True
        st.session_state["pending_auto_analyze"] = True
        st.rerun()


def show_landing_page():
    """Render the Market Pulse landing page."""
    # --- Hero header ---
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <h1 style="font-size:2.5rem; font-weight:800; margin:0;">
            📊 PRISM <span style="color:#D4A017;">v2.1</span>
        </h1>
        <p style="color:#888; font-size:1.1rem; margin-top:0.5rem;">
            Quantitative Stock Analysis · 9-Layer Framework
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Search bar (redundant with sidebar but visible on landing) ---
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        quick_ticker = st.text_input(
            "Search Stock",
            placeholder="e.g. AAPL, PTT.BK, TSLA...",
            label_visibility="collapsed",
            key="landing_search",
        )
    with col_btn:
        if st.button("🔍 Analyze", width="stretch"):
            if quick_ticker.strip():
                st.session_state["selected_ticker"] = quick_ticker.strip().upper()
                st.session_state["run_analysis"] = True
                st.session_state["pending_auto_analyze"] = True
                st.rerun()

    st.divider()

    # --- Market Pulse tabs ---
    tab_names = list(WATCHLIST.keys())
    tabs = st.tabs(tab_names)

    for tab, (section_name, stocks) in zip(tabs, WATCHLIST.items()):
        with tab:
            cols = st.columns(4)  # 4 cards per row (2 rows for 8 stocks)
            for i, stock in enumerate(stocks):
                with cols[i % 4]:
                    data = fetch_card_data(stock["ticker"])
                    _render_pulse_card(stock["ticker"], stock["name"], data)
