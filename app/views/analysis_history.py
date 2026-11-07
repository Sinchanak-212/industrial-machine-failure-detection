"""
AURALIS Analysis History View.
Provides a comprehensive audit trail of all acoustic analyses executed during the active session.
Includes section and status filtering, aggregate telemetry, and CSV log export.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from components.cards import render_telemetry_card
from components.icons import get_svg_icon


def render_analysis_history(history: list, on_navigate_to_analysis=None):
    """
    Renders the Analysis History page.
    """
    st.markdown("### Acoustic Diagnostic Audit Trail")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:18px; font-size:0.88rem;'>"
        "Chronological audit log of all acoustic recordings analyzed during the active monitoring session. "
        "Filter by machine unit or diagnosis, inspect scores, and export session records."
        "</p>",
        unsafe_allow_html=True
    )

    if not history:
        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px dashed #1E2638; border-radius:8px; padding:40px 20px; text-align:center;">
                <div style="color:#06B6D4; margin-bottom:10px;">{get_svg_icon('history', 36, '#06B6D4')}</div>
                <div style="font-size:1rem; font-weight:600; color:#F8FAFC;">No Diagnostic Records in Active Session</div>
                <div style="font-size:0.82rem; color:#94A3B8; max-width:480px; margin:6px auto 16px auto; line-height:1.5;">
                    Records are dynamically appended to this audit trail whenever an audio file is evaluated in <b>Live Analysis</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if on_navigate_to_analysis:
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c2:
                if st.button("Proceed to Live Analysis", type="primary", use_container_width=True):
                    on_navigate_to_analysis()
                    st.rerun()
        return

    # Calculate Session Statistics
    total_count = len(history)
    normal_count = sum(1 for h in history if not h.get("is_anomaly", False))
    anomaly_count = sum(1 for h in history if h.get("is_anomaly", False))
    scores = [h.get("anomaly_score", 0.0) for h in history]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_telemetry_card(
            label="Total Analyzed",
            value=str(total_count),
            subtext="Recordings in session",
            icon_name="activity",
            accent="cyan"
        )
    with col2:
        render_telemetry_card(
            label="Normal Diagnoses",
            value=f"{normal_count}",
            subtext=f"{(normal_count/total_count*100):.1f}% of tests",
            icon_name="check_circle",
            accent="cyan"
        )
    with col3:
        render_telemetry_card(
            label="Anomalous Alerts",
            value=f"{anomaly_count}",
            subtext=f"{(anomaly_count/total_count*100):.1f}% of tests",
            icon_name="alert_triangle",
            accent="red" if anomaly_count > 0 else "default"
        )
    with col4:
        render_telemetry_card(
            label="Mean Reconstruction Loss",
            value=f"{mean_score:.6f}",
            subtext="Average MSE across session",
            icon_name="performance",
            accent="amber"
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Filter Controls
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        section_filter = st.selectbox(
            "Filter by Machine Section",
            options=["All Sections", "section_00", "section_01", "section_02"]
        )
    with col_f2:
        status_filter = st.selectbox(
            "Filter by Diagnostic Verdict",
            options=["All Diagnoses", "Normal Only", "Anomalous Only"]
        )

    # Apply Filters
    filtered_history = history
    if section_filter != "All Sections":
        filtered_history = [h for h in filtered_history if h.get("section") == section_filter]
    if status_filter == "Normal Only":
        filtered_history = [h for h in filtered_history if not h.get("is_anomaly", False)]
    elif status_filter == "Anomalous Only":
        filtered_history = [h for h in filtered_history if h.get("is_anomaly", False)]

    st.markdown(f"##### Showing {len(filtered_history)} of {total_count} Diagnostic Records")

    if not filtered_history:
        st.info("No records match the selected filter criteria.")
    else:
        df_display = pd.DataFrame(filtered_history)
        cols_map = {
            "timestamp": "Timestamp",
            "filename": "Audio Recording",
            "section": "Machine Section",
            "anomaly_score": "MSE Anomaly Score",
            "threshold": "Validation Threshold",
            "prediction": "Diagnosis",
            "device": "Compute Hardware",
            "latency_ms": "Latency (ms)"
        }
        cols_present = [c for c in cols_map.keys() if c in df_display.columns]
        
        st.dataframe(
            df_display[cols_present].rename(columns=cols_map),
            use_container_width=True,
            hide_index=True
        )

        col_dl1, col_dl2 = st.columns([1, 2])
        with col_dl1:
            csv_data = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Session Audit Trail (CSV)",
                data=csv_data,
                file_name=f"auralis_audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

