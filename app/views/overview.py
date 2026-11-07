"""
AURALIS Overview View.
Provides industrial fleet telemetry, active machine section monitoring,
latest analysis status, model/device telemetry, and recent session audit feed.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from components.cards import render_telemetry_card, render_scientific_callout
from components.icons import get_svg_icon
from inference import THRESHOLDS, DEVICE


def render_overview(history: list, current_section: str, on_navigate_to_analysis=None):
    """
    Renders the AURALIS Overview page.
    """
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("### Fleet Acoustic Monitoring & Machine Intelligence")
        st.markdown(
            "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:18px; font-size:0.88rem;'>"
            "Continuous non-contact acoustic surveillance for industrial ventilation fans. "
            "Unsupervised condition monitoring powered by section-specific PyTorch CNN Autoencoders."
            "</p>",
            unsafe_allow_html=True
        )
    with col_t2:
        if st.button("Launch Live Diagnostic", type="primary", use_container_width=True):
            if on_navigate_to_analysis:
                on_navigate_to_analysis()
                st.rerun()

    # Session Quantitative Telemetry
    total_analyzed = len(history)
    normal_count = sum(1 for item in history if not item.get("is_anomaly", False))
    anomaly_count = sum(1 for item in history if item.get("is_anomaly", False))
    
    normal_pct = f"{(normal_count / total_analyzed * 100):.1f}%" if total_analyzed > 0 else "0.0%"
    anomaly_pct = f"{(anomaly_count / total_analyzed * 100):.1f}%" if total_analyzed > 0 else "0.0%"
    active_threshold = THRESHOLDS.get(current_section, 0.0)

    # 4 Industrial Telemetry KPI Tiles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_telemetry_card(
            label="Recordings Analyzed",
            value=str(total_analyzed),
            subtext="Session diagnostics executed",
            icon_name="activity",
            accent="cyan"
        )
    with col2:
        render_telemetry_card(
            label="Normal Baseline",
            value=f"{normal_count}",
            subtext=f"{normal_pct} within tolerance",
            icon_name="check_circle",
            accent="cyan"
        )
    with col3:
        render_telemetry_card(
            label="Anomalous Alerts",
            value=f"{anomaly_count}",
            subtext=f"{anomaly_pct} exceeding cutoff",
            icon_name="alert_triangle",
            accent="red" if anomaly_count > 0 else "default"
        )
    with col4:
        render_telemetry_card(
            label="Calibrated Threshold",
            value=f"{active_threshold:.6f}",
            subtext=f"{current_section} (P95 normal cutoff)",
            icon_name="shield",
            accent="amber"
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Latest Analysis Status (if an analysis has run in this session)
    if "last_result" in st.session_state and st.session_state["last_result"] is not None:
        last_rec = st.session_state["last_result"]["record"]
        is_anom = last_rec["is_anomaly"]
        status_color = "#EF4444" if is_anom else "#06B6D4"
        status_text = "ANOMALY ALERT" if is_anom else "NORMAL OPERATION"

        st.markdown(
            f"""
            <div style="background-color:#0E131F; border:1px solid {'rgba(239, 68, 68, 0.35)' if is_anom else 'rgba(6, 182, 212, 0.35)'}; border-radius:8px; padding:16px 20px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">LATEST ANALYSIS RESULT:</span>
                        <span class="telemetry-badge {'badge-red' if is_anom else 'badge-cyan'}">{status_text}</span>
                        <span style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#CBD5E1;">{last_rec['filename']}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#94A3B8;">
                        MSE: <strong style="color:{status_color};">{last_rec['anomaly_score']:.6f}</strong> &bull; 
                        THRESHOLD: <strong style="color:#F59E0B;">{last_rec['threshold']:.6f}</strong> &bull;
                        MACHINE: <strong style="color:#38BDF8;">{last_rec['section']}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Recent Session Audit Stream
    st.markdown("#### Session Audit Stream")

    if total_analyzed == 0:
        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px dashed #1E2638; border-radius:8px; padding:32px 20px; text-align:center;">
                <div style="color:#06B6D4; margin-bottom:8px;">{get_svg_icon('activity', 32, '#06B6D4')}</div>
                <div style="font-size:0.95rem; font-weight:600; color:#F8FAFC;">No Recordings Analyzed in this Session</div>
                <div style="font-size:0.82rem; color:#94A3B8; max-width:500px; margin:6px auto 16px auto; line-height:1.5;">
                    Live telemetry metrics update as diagnostic audio streams are processed. 
                    Navigate to <b>Live Analysis</b> to upload a recording or evaluate a benchmark sound sample.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        df_history = pd.DataFrame(history)
        cols_to_show = ["timestamp", "filename", "section", "anomaly_score", "threshold", "prediction", "device"]
        available_cols = [c for c in cols_to_show if c in df_history.columns]
        
        st.dataframe(
            df_history[available_cols].rename(columns={
                "timestamp": "Timestamp",
                "filename": "Audio Recording",
                "section": "Machine Section",
                "anomaly_score": "MSE Score",
                "threshold": "Calibrated Threshold",
                "prediction": "Diagnosis",
                "device": "Compute Hardware"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Operational Specifications Grid
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### Machine Section Threshold Matrix")
        st.markdown(
            """
            <table style="width:100%; border-collapse:collapse; font-size:0.82rem; font-family:'JetBrains Mono', monospace; background:#0E131F; border:1px solid #1E2638; border-radius:6px; overflow:hidden;">
                <thead>
                    <tr style="background:#131929; color:#06B6D4; text-align:left; border-bottom:1px solid #1E2638;">
                        <th style="padding:8px 12px;">Machine Unit</th>
                        <th style="padding:8px 12px;">Calibrated P95 Threshold</th>
                        <th style="padding:8px 12px;">Validation N</th>
                        <th style="padding:8px 12px;">Observed FPR</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #1E2638;">
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_00</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00347179</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 normal</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1E2638;">
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_01</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00448493</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 normal</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_02</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00319858</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 normal</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                    </tr>
                </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )

    with col_r:
        st.markdown("#### Hardware & Topology Summary")
        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:16px; font-size:0.84rem; color:#CBD5E1; line-height:1.6;">
                <div style="font-weight:600; color:#06B6D4; margin-bottom:6px;">Inference Execution Pipeline</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#94A3B8;">Compute Hardware:</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:600; color:#F8FAFC;">{str(DEVICE).upper()}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#94A3B8;">Neural Architecture:</span>
                    <span style="font-family:'JetBrains Mono'; color:#F8FAFC;">3-Stage Conv2d Autoencoder</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span style="color:#94A3B8;">Acoustic Features:</span>
                    <span style="font-family:'JetBrains Mono'; color:#F8FAFC;">40 MFCCs (16 kHz, N_FFT=1024, Hop=256)</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#94A3B8;">Weights Status:</span>
                    <span class="telemetry-badge badge-green" style="font-size:0.65rem;">FROZEN VALIDATION LOCKED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Scientific Notice Callout
    render_scientific_callout()
