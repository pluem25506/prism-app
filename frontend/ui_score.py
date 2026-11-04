import math

import streamlit as st

LAYER_INFO = {
    "S1": ("L1", "Quality Screen", 0.20),
    "S2": ("L2", "Value Creation", 0.15),
    "S3": ("L3", "Valuation", 0.20),
    "S4": ("L4", "Risk & Leverage", 0.15),
    "S5": ("L5", "Monte Carlo", 0.10),
    "S6": ("L6", "Time-Series", 0.05),
    "S7": ("L7", "Portfolio Fit", 0.05),
    "S8": ("L8", "Op./Fin. Leverage", 0.05),
    "S9": ("L9", "Contextual Intel", 0.05),
}

_GREEN = "#00C896"
_YELLOW = "#FFA502"
_RED = "#FF4757"
_BORDER = "#1E3A5F"
_BG = "#0A1628"
_TEXT_SECONDARY = "#8899AA"
_GOLD = "#D4A017"
_MUTED = "#3A4A5E"


def _score_color(score: float) -> str:
    # Color logic -- MUST match v2.1 spec (>=70 GREEN, >=45 YELLOW, <45 RED)
    if score >= 70:
        return _GREEN
    if score >= 45:
        return _YELLOW
    return _RED


def render_score_gauge(score: float, signal: str):
    """Render a filled-donut-segment speedometer gauge for PRISM Score 0-100."""
    if signal == "BUY":
        color = "#00C896"
        label = "BUY"
    elif signal == "HOLD":
        color = "#D4A017"
        label = "HOLD"
    elif signal == "MIXED SIGNAL":
        # Not a SELL -- valuation models disagree, so give it its own neutral
        # color rather than falsely painting it red.
        color = "#8899AA"
        label = "MIXED SIGNAL"
    else:
        color = "#FF4757"
        label = "SELL"

    CX, CY = 170, 170
    R_OUTER, R_INNER = 130, 95

    def to_rad(deg):
        return deg * math.pi / 180

    def score_to_angle(s):
        # score 0 -> 180deg, score 100 -> 0deg
        return 180 - (s / 100) * 180

    def polar(angle_deg, r):
        a = to_rad(angle_deg)
        return CX + r * math.cos(a), CY - r * math.sin(a)

    def segment_path(start_s, end_s):
        a1 = score_to_angle(start_s)
        a2 = score_to_angle(end_s)
        o1x, o1y = polar(a1, R_OUTER)
        o2x, o2y = polar(a2, R_OUTER)
        i1x, i1y = polar(a1, R_INNER)
        i2x, i2y = polar(a2, R_INNER)
        large = 1 if (end_s - start_s) > 50 else 0
        return (
            f"M {o1x:.2f} {o1y:.2f} "
            f"A {R_OUTER} {R_OUTER} 0 {large} 1 {o2x:.2f} {o2y:.2f} "
            f"L {i2x:.2f} {i2y:.2f} "
            f"A {R_INNER} {R_INNER} 0 {large} 0 {i1x:.2f} {i1y:.2f} Z"
        )

    # Segment paths
    segs = [
        (segment_path(0, 45), "#FF4757"),
        (segment_path(45, 65), "#D4A017"),
        (segment_path(65, 100), "#00C896"),
    ]
    seg_svg = "".join(f'<path d="{d}" fill="{c}" opacity="0.92"/>' for d, c in segs)

    # Divider lines at boundaries
    dividers = ""
    for boundary in [45, 65]:
        a = score_to_angle(boundary)
        x1, y1 = polar(a, R_INNER)
        x2, y2 = polar(a, R_OUTER)
        dividers += (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#0A1628" stroke-width="3"/>'
        )

    # Labels centered in each segment (outside arc)
    label_r = R_OUTER + 20
    label_items = [
        (22, "SELL", "#FF4757"),
        (55, "HOLD", "#D4A017"),
        (82, "BUY", "#00C896"),
    ]
    labels_svg = ""
    for s, txt, c in label_items:
        a = score_to_angle(s)
        lx, ly = polar(a, label_r)
        labels_svg += (
            f'<text x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="800" fill="{c}" '
            f'font-family="sans-serif">{txt}</text>'
        )

    # Needle (triangle)
    R_MID = (R_OUTER + R_INNER) / 2
    needle_angle = score_to_angle(score)
    tip_x, tip_y = polar(needle_angle, R_MID + 10)
    base1_x, base1_y = polar(needle_angle + 90, 8)
    base2_x, base2_y = polar(needle_angle - 90, 8)
    needle_svg = (
        f'<polygon points="{tip_x:.1f},{tip_y:.1f} '
        f'{base1_x:.1f},{base1_y:.1f} {base2_x:.1f},{base2_y:.1f}" '
        'fill="white" opacity="0.95"/>'
        f'<circle cx="{CX}" cy="{CY}" r="8" fill="white"/>'
    )

    # Built as a single-line string on purpose: a multi-line, Python-indented
    # f-string here gets read by Streamlit's markdown parser as an indented
    # code block, rendering the raw "<svg...>" text instead of the chart.
    gauge_svg = (
        '<svg width="340" height="180" viewBox="0 0 340 200" '
        'xmlns="http://www.w3.org/2000/svg">'
        f"{seg_svg}{dividers}{labels_svg}{needle_svg}"
        "</svg>"
    )

    st.markdown(f"""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
        {gauge_svg}
        <div style="font-size:3.5rem; font-weight:900; color:white;
                    line-height:1; margin-top:0.3rem;">{score:.0f}</div>
        <div style="font-size:0.7rem; letter-spacing:0.2em; color:#556;
                    text-transform:uppercase; margin-top:4px;">PRISM SCORE</div>
        <div style="font-size:1.5rem; font-weight:800; color:{color};
                    margin-top:0.6rem; letter-spacing:0.05em;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(result: dict):
    score = result["prism_score"]
    signal = result["signal"]
    confidence = result["confidence"]

    render_score_gauge(score, signal)

    # Sub-scores -- weights MUST match v2.1 spec exactly
    rows_html = []
    for key, (layer, name, weight) in LAYER_INFO.items():
        sub = result["sub_scores"].get(key)
        if sub is not None:
            row_color = _score_color(sub)
            score_label = f"{sub:.0f}"
            bar_width = max(0.0, min(100.0, sub))
        else:
            row_color = _MUTED
            score_label = "N/A"
            bar_width = 0.0

        rows_html.append(
            f"""
            <div style="display:flex; align-items:center;
                        padding:8px 0; border-bottom:1px solid {_BORDER};">
              <span style="color:{_TEXT_SECONDARY}; width:30px; font-size:12px">{layer}</span>
              <span style="color:#FFFFFF; flex:1; font-size:13px">{name}</span>
              <div style="width:100px; background:{_BG};
                          border-radius:4px; height:6px; margin-right:12px;">
                <div style="width:{bar_width:.0f}%; background:{row_color};
                            height:6px; border-radius:4px;"></div>
              </div>
              <span style="color:{row_color}; width:35px;
                           text-align:right; font-size:14px; font-weight:600">
                {score_label}
              </span>
              <span style="color:{_TEXT_SECONDARY}; width:40px;
                           text-align:right; font-size:12px">{weight*100:.0f}%</span>
            </div>
            """
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)

    # Confidence meter
    st.markdown(
        f"""
        <div style="margin:16px 0;">
          <div style="display:flex; justify-content:space-between;
                      margin-bottom:6px;">
            <span style="color:{_TEXT_SECONDARY}; font-size:12px">DATA CONFIDENCE</span>
            <span style="color:{_GOLD}; font-size:12px;
                         font-weight:600">{confidence:.0f}%</span>
          </div>
          <div style="background:{_BG}; border-radius:4px; height:6px;">
            <div style="width:{confidence:.0f}%; background:linear-gradient(
                        90deg, {_GOLD}, #FFD700);
                        height:6px; border-radius:4px;"></div>
          </div>
          <div style="color:{_TEXT_SECONDARY}; font-size:11px; margin-top:4px">
            {result['available_layers']}/9 layers computed
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mixed signal warning
    if signal == "MIXED SIGNAL":
        st.warning("Valuation models disagree — exercise additional due diligence before deciding.")
