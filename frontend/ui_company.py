import html

import streamlit as st

from backend.data_fetcher import get_currency, normalize_dividend_yield

_CARD_BG = "#0F1F3D"
_BORDER = "#1E3A5F"
_BG = "#0A1628"
_TEXT_SECONDARY = "#8899AA"
_GOLD = "#D4A017"

_SECTOR_PILL_STYLES = {
    "Financial": ("#1A3A5C", "#60A5FA"),  # blue
    "Energy": ("#3D2000", "#FFA502"),  # amber
    "Technology": ("#1A0040", "#A78BFA"),  # purple
}
_DEFAULT_PILL_STYLE = ("#1E3A5F", "#8899AA")  # gray

_COUNTRY_FLAGS = {
    "Thailand": "🇹🇭",
    "United States": "🇺🇸",
}
_DEFAULT_FLAG = "🌍"


def format_market_cap(value, symbol: str = "฿") -> str:
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{symbol}{value/1e12:.1f}T"
    if value >= 1e9:
        return f"{symbol}{value/1e9:.1f}B"
    if value >= 1e6:
        return f"{symbol}{value/1e6:.1f}M"
    return "N/A"


def _sector_pill_style(sector: str):
    for keyword, style in _SECTOR_PILL_STYLES.items():
        if keyword in sector:
            return style
    return _DEFAULT_PILL_STYLE


def render_company_profile(ticker: str, info: dict):
    """Company profile card shown at the top of the main area, above the
    PRISM score card."""
    info = info or {}

    name = info.get("longName", ticker)
    sector = info.get("sector", "N/A") or "N/A"
    industry = info.get("industry", "N/A") or "N/A"
    country = info.get("country", "N/A") or "N/A"
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    week52_high = info.get("fiftyTwoWeekHigh")
    week52_low = info.get("fiftyTwoWeekLow")
    div_yield = normalize_dividend_yield(info.get("dividendYield"), info.get("dividendRate"), price)
    employees = info.get("fullTimeEmployees")
    description = (info.get("longBusinessSummary") or "")[:300]

    currency, symbol = get_currency(ticker)

    price_str = f"{symbol}{price:.2f}" if price is not None else "N/A"
    market_cap_str = format_market_cap(market_cap, symbol)
    if week52_low is not None and week52_high is not None:
        range_str = f"{symbol}{week52_low:.2f} - {symbol}{week52_high:.2f}"
    else:
        range_str = "N/A"
    yield_str = f"{div_yield * 100:.2f}%" if div_yield else "No dividend"

    pill_bg, pill_color = _sector_pill_style(sector)
    flag = _COUNTRY_FLAGS.get(country, _DEFAULT_FLAG)

    # yfinance text fields (longName, sector, etc.) can contain "&", "<" etc.
    # (e.g. "AT&T Inc.") -- escape before embedding in raw HTML.
    name_esc = html.escape(name)
    sector_esc = html.escape(sector)
    industry_esc = html.escape(industry)
    country_esc = html.escape(country)

    # 52-week range bar: current price's position between the yearly low/high
    range_bar_html = ""
    if price is not None and week52_low is not None and week52_high is not None and week52_high > week52_low:
        pct = max(0.0, min(100.0, (price - week52_low) / (week52_high - week52_low) * 100))
        range_bar_html = f"""
        <div style="margin-top:16px;">
          <div style="display:flex; justify-content:space-between; font-size:11px;
                      color:{_TEXT_SECONDARY}; margin-bottom:6px;">
            <span>{symbol}{week52_low:.2f}</span>
            <span style="color:{_GOLD}; font-weight:600;">&#9679; {symbol}{price:.2f}</span>
            <span>{symbol}{week52_high:.2f}</span>
          </div>
          <div style="position:relative; background:{_BG}; border-radius:4px; height:6px;">
            <div style="position:absolute; left:{pct:.1f}%; top:-3px; width:12px; height:12px;
                        background:{_GOLD}; border-radius:50%; transform:translateX(-50%);
                        border:2px solid {_CARD_BG};"></div>
          </div>
        </div>
        """

    st.markdown(
        f"""
        <div style="background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:12px;
                    padding:20px; margin-bottom:20px;">
          <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px; flex-wrap:wrap;">
            <span style="font-size:20px; font-weight:700; color:#FFFFFF;">{name_esc}</span>
            <span style="background:{pill_bg}; color:{pill_color}; padding:4px 12px;
                         border-radius:12px; font-size:12px; font-weight:600;">{sector_esc}</span>
            <span style="font-size:14px; color:{_TEXT_SECONDARY};">{flag} {country_esc}</span>
          </div>
          <div style="color:{_TEXT_SECONDARY}; font-size:13px; margin-bottom:16px;">{industry_esc}</div>

          <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <div style="flex:1; min-width:120px; background:{_BG}; border-radius:8px; padding:12px;">
              <div style="color:{_TEXT_SECONDARY}; font-size:11px;">CURRENT PRICE ({currency})</div>
              <div style="color:#FFFFFF; font-size:20px; font-weight:600;">{price_str}</div>
            </div>
            <div style="flex:1; min-width:120px; background:{_BG}; border-radius:8px; padding:12px;">
              <div style="color:{_TEXT_SECONDARY}; font-size:11px;">MARKET CAP</div>
              <div style="color:#FFFFFF; font-size:20px; font-weight:600;">{market_cap_str}</div>
            </div>
            <div style="flex:1; min-width:120px; background:{_BG}; border-radius:8px; padding:12px;">
              <div style="color:{_TEXT_SECONDARY}; font-size:11px;">52W RANGE</div>
              <div style="color:#FFFFFF; font-size:20px; font-weight:600;">{range_str}</div>
            </div>
            <div style="flex:1; min-width:120px; background:{_BG}; border-radius:8px; padding:12px;">
              <div style="color:{_TEXT_SECONDARY}; font-size:11px;">DIVIDEND YIELD</div>
              <div style="color:#FFFFFF; font-size:20px; font-weight:600;">{yield_str}</div>
            </div>
          </div>
          {range_bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(f"About {name}", expanded=False):
        st.caption(f"{description}..." if description else "No description available.")
        if employees:
            st.caption(f"Employees: {employees:,}")
