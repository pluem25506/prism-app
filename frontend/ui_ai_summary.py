"""
AI Summary panel: a Gemini-generated plain-language readout of the PRISM
quantitative output. Math computes every number here; the AI only explains
results already produced by the backend -- it never recomputes or overrides
anything.
"""

import html

import streamlit as st
from google import genai
from google.genai import types

_SYSTEM_PROMPT_EN = """You are a professional investment analyst.
Summarize this quantitative analysis in 4-5 sentences.
Cover: key strengths, key risks, and what investors should watch.
Do NOT give direct buy/sell advice -- this is quantitative analysis only."""

_SYSTEM_PROMPT_TH = """คุณเป็นนักวิเคราะห์การลงทุนมืออาชีพ
สรุปผลการวิเคราะห์เชิงปริมาณของหุ้นนี้เป็นภาษาไทย
ใน 4-5 ประโยค เน้นจุดแข็ง จุดอ่อน และสิ่งที่นักลงทุนควรระวัง
ห้ามแนะนำให้ซื้อหรือขายโดยตรง เพราะนี่คือการวิเคราะห์เชิงปริมาณเท่านั้น"""


def _build_prompt(result: dict, lang: str) -> str:
    ticker = result["ticker"]
    score = result["prism_score"]
    signal = result["signal"]
    conf = result["confidence"]
    subs = result["sub_scores"]
    l4 = result["layer_details"].get("L4", {})
    l5 = result["layer_details"].get("L5", {})

    data_str = f"""
Stock: {ticker}
PRISM Score: {score:.1f}/100
Signal: {signal}
Confidence: {conf:.0f}%

Layer Scores:
- L1 Quality Screen: {subs.get('S1', 'N/A')}
- L2 Value Creation: {subs.get('S2', 'N/A')}
- L3 Valuation: {subs.get('S3', 'N/A')}
- L4 Risk: {subs.get('S4', 'N/A')}
- L5 Monte Carlo: {subs.get('S5', 'N/A')}

Risk Metrics:
- Sharpe Ratio: {l4.get('sharpe', 'N/A')}
- Beta: {l4.get('beta', 'N/A')}
- VaR 95%: {l4.get('VaR_95', 'N/A')}
- Monte Carlo P50 upside: {l5.get('upside_P50', 'N/A')}
- Prob. of loss: {l5.get('prob_loss', 'N/A')}
"""

    system = _SYSTEM_PROMPT_TH if lang == "th" else _SYSTEM_PROMPT_EN
    return system + "\n\n" + data_str


@st.cache_data(ttl=300, show_spinner=False)
def _generate_summary(prompt: str) -> str:
    """Cached Gemini call, keyed on the exact prompt text -- a rerender with
    the same ticker/score/signal/language is a cache hit and skips the API."""
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        # "gemini-1.5-flash" (the originally specified model) and every
        # subsequent "flash"/"flash-lite" tier we tried through it have been
        # retired or overloaded. gemini-3.1-flash-lite on the current
        # google.genai SDK, with thinking disabled, responds in ~2s --
        # matching what a "flash" model should feel like. See chat for the
        # full investigation (model retirements, 503 overload, SDK swap).
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=30000),
            ),
        )
        return response.text
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"


def render_ai_summary(result: dict, lang: str = "en"):
    st.markdown("### 🤖 AI Summary")

    col1, col2 = st.columns([1, 4])
    with col1:
        lang_choice = st.radio("Language", ["EN", "TH"], index=1 if lang == "th" else 0, horizontal=True)

    chosen_lang = "th" if lang_choice == "TH" else "en"
    prompt = _build_prompt(result, chosen_lang)

    with st.spinner("Generating AI summary..."):
        summary_text = _generate_summary(prompt)

    # LLM output is freeform text -- escape before embedding in raw HTML,
    # and preserve its paragraph breaks (HTML collapses bare newlines).
    summary_html = html.escape(summary_text).replace("\n", "<br>")

    st.markdown(
        f"""
        <div style="background:#0F1F3D; border-left:3px solid #D4A017;
                    border-radius:0 8px 8px 0; padding:16px 20px;
                    margin:12px 0;">
          <div style="color:#D4A017; font-size:11px;
                      letter-spacing:1px; margin-bottom:8px">
            🤖 AI QUANTITATIVE SUMMARY
          </div>
          <div style="color:#FFFFFF; font-size:14px; line-height:1.7">
            {summary_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("⚠️ AI summary reads quantitative output only. Not investment advice.")
