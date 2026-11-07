"""
AURALIS Card Components.
Provides KPI telemetry tiles, industrial diagnostic result banners, and scientific notice callouts.
Uses precision vector SVG icons instead of emojis.
"""

import streamlit as st
from components.icons import get_svg_icon


def render_telemetry_card(
    label: str,
    value: str,
    subtext: str = "",
    icon_name: str = "activity",
    accent: str = "default"
):
    """
    Renders an industrial telemetry card.
    accent: 'default' | 'cyan' | 'green' | 'amber' | 'red'
    """
    icon_color = "#94A3B8"
    if accent == "cyan":
        icon_color = "#06B6D4"
    elif accent == "green":
        icon_color = "#10B981"
    elif accent == "amber":
        icon_color = "#F59E0B"
    elif accent == "red":
        icon_color = "#EF4444"

    svg_icon = get_svg_icon(icon_name, size=16, color=icon_color)

    html = f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-label">{label}</span>
            <div class="kpi-icon-container">{svg_icon}</div>
        </div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtext">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_diagnostic_banner(
    prediction: str,
    anomaly_score: float,
    threshold: float,
    filename: str,
    device: str = "cuda"
):
    """
    Renders an industrial verdict banner for audio analysis.
    """
    is_anomaly = prediction.lower() == "anomalous"
    status_class = "anomaly" if is_anomaly else "normal"
    
    icon_svg = get_svg_icon("alert_triangle", 20, "#EF4444") if is_anomaly else get_svg_icon("check_circle", 20, "#06B6D4")
    title_text = "ANOMALOUS ACOUSTIC PATTERN DETECTED" if is_anomaly else "NORMAL ACOUSTIC PATTERN CONFIRMED"
    
    score_ratio = (anomaly_score / threshold) if threshold > 0 else 1.0
    delta = anomaly_score - threshold
    delta_sign = "+" if delta > 0 else ""
    
    if is_anomaly:
        desc_text = (
            f"Autoencoder reconstruction error (MSE: <code>{anomaly_score:.6f}</code>) exceeds the locked validation threshold "
            f"(<code>{threshold:.6f}</code>) by <b>{delta_sign}{delta:.6f}</b> ({score_ratio:.2f}&times; threshold). "
            f"Acoustic spectral distribution indicates mechanical degradation or operational irregularity."
        )
    else:
        desc_text = (
            f"Autoencoder reconstruction error (MSE: <code>{anomaly_score:.6f}</code>) is within the nominal operating baseline "
            f"below the locked validation threshold (<code>{threshold:.6f}</code>) with safety margin <b>{abs(delta):.6f}</b> "
            f"({score_ratio:.2f}&times; threshold). Sound profile matches healthy fan acoustics."
        )

    html = f"""
    <div class="industrial-verdict {status_class}">
        <div style="flex: 1;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                <div class="verdict-title {status_class}">
                    <span>{icon_svg}</span>
                    <span>{title_text}</span>
                </div>
                <div style="display:flex; gap:8px;">
                    <span class="telemetry-badge {'badge-red' if is_anomaly else 'badge-cyan'}">
                        VERDICT: {prediction.upper()}
                    </span>
                    <span class="telemetry-badge badge-amber">
                        RATIO: {score_ratio:.2f}X
                    </span>
                </div>
            </div>
            <div class="verdict-desc">
                {desc_text}
            </div>
            <div style="margin-top:12px; font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#64748B; display:flex; gap:16px; flex-wrap:wrap;">
                <span>RECORDING: <strong style="color:#94A3B8;">{filename}</strong></span>
                <span>DEVICE: <strong style="color:#94A3B8;">{device.upper()}</strong></span>
                <span>ALGORITHM: <strong style="color:#94A3B8;">MFCC CNN Autoencoder</strong></span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_scientific_callout():
    """
    Renders the scientific notice disclosing prototype boundaries.
    """
    icon_svg = get_svg_icon("alert_triangle", 16, "#F59E0B")
    st.markdown(
        f"""
        <div class="audit-notice amber">
            <div style="font-weight:600; color:#F59E0B; margin-bottom:4px; display:flex; align-items:center; gap:8px;">
                <span>{icon_svg}</span>
                <span>RESEARCH & SYSTEM SPECIFICATION NOTICE</span>
            </div>
            <div>
                This system operates as an <b>acoustic condition-monitoring prototype</b> trained exclusively on normal operational sounds. 
                In rigorous held-out target testing, it achieved a <b>Macro ROC-AUC of 0.6091</b> [95% CI: 0.5651–0.6543] with a 
                <b>Macro Recall of 5.67%</b> at the locked 95th-percentile validation threshold. 
                It is an <b>experimental acoustic anomaly detector</b>, not a field-validated predictive maintenance tool.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
