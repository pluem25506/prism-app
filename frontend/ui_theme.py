"""Shared styling tweaks applied on every page (app.py and pages/backtest.py
are separate script executions in Streamlit's multipage model, so each
page calls this itself rather than it running once centrally)."""

import streamlit as st

_GOLD = "#D4A017"
_TEXT_SECONDARY = "#8899AA"
_BORDER = "#1E3A5F"


def inject_sidebar_nav_style():
    """Make Streamlit's auto-generated "app"/"backtest" page-nav list (the
    links at the very top of the sidebar, from the pages/ folder) readable --
    by default it's small, low-contrast grey text that's easy to miss
    entirely against the dark sidebar background."""
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebarNav"] {{
            padding-top: 0.5rem;
            padding-bottom: 0.75rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid {_BORDER};
        }}
        [data-testid="stSidebarNavLink"] {{
            border-radius: 6px;
            margin: 2px 0;
        }}
        [data-testid="stSidebarNavLink"] p {{
            font-size: 15px !important;
            font-weight: 700 !important;
            color: {_TEXT_SECONDARY} !important;
            text-transform: capitalize;
        }}
        [data-testid="stSidebarNavLink"]:hover p {{
            color: #FFFFFF !important;
        }}
        [data-testid="stSidebarNavLink"][aria-current="page"] {{
            background-color: rgba(212, 160, 23, 0.15) !important;
        }}
        [data-testid="stSidebarNavLink"][aria-current="page"] p {{
            color: {_GOLD} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
