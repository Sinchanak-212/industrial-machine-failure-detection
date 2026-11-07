"""
AURALIS | Industrial Intelligence.
Enterprise-grade industrial condition-monitoring dashboard.
Integrates PyTorch CNN Autoencoder backend with Streamlit.
"""

import os
from pathlib import Path
import sys

# Ensure root directory is set as working directory and added to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "app") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "app"))

import streamlit as st

# Configure wide page layout and industrial branding
st.set_page_config(
    page_title="AURALIS | Industrial Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply enterprise industrial dark theme CSS
from styles import AURALIS_CSS
st.markdown(AURALIS_CSS, unsafe_allow_html=True)

from components.header import render_industrial_header
from components.icons import get_svg_icon
from inference import THRESHOLDS, DEVICE
from views.overview import render_overview
from views.audio_analysis import render_audio_analysis
from views.analysis_history import render_analysis_history
from views.model_info import render_model_info
from views.research_performance import render_research_performance
from views.about import render_about


def init_session_state():
    """Initializes app session state variables."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "selected_section" not in st.session_state:
        st.session_state["selected_section"] = "section_00"
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "Overview"
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None


def update_section(new_sec: str):
    """Callback to update active machine section across app."""
    st.session_state["selected_section"] = new_sec


def main():
    init_session_state()

    current_sec = st.session_state["selected_section"]

    # ==========================
    # SIDEBAR
    # ==========================
    with st.sidebar:
        # AURALIS Brand Identity
        logo_svg = get_svg_icon("logo", 22, "#06B6D4")
        st.markdown(
            f"""
            <div style="padding: 12px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 16px;">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="background-color: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.35); width:36px; height:36px; border-radius:6px; display:flex; align-items:center; justify-content:center;">
                        {logo_svg}
                    </div>
                    <div>
                        <div style="font-weight:700; font-size:1.1rem; color:#F8FAFC; letter-spacing:0.02em;">AURALIS</div>
                        <div style="font-size:0.7rem; color:#06B6D4; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">Industrial Intelligence</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Global Machine Section Selector
        st.markdown(
            "<div style='font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;'>Target Asset Section</div>",
            unsafe_allow_html=True
        )
        sections = list(THRESHOLDS.keys())
        active_idx = sections.index(current_sec) if current_sec in sections else 0
        
        sidebar_sec = st.selectbox(
            "Machine Section",
            options=sections,
            index=active_idx,
            label_visibility="collapsed",
            key="sidebar_section_selector"
        )
        if sidebar_sec != current_sec:
            st.session_state["selected_section"] = sidebar_sec
            current_sec = sidebar_sec

        active_thr = THRESHOLDS[current_sec]
        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px solid #182236; border-radius:4px; padding:6px 10px; margin-top:2px; margin-bottom:16px; font-size:0.72rem; font-family:'JetBrains Mono', monospace; color:#94A3B8; display:flex; justify-content:space-between;">
                <span>Locked P95:</span>
                <span style="color:#F59E0B; font-weight:600;">{active_thr:.6f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Navigation Menu
        st.markdown(
            "<div style='font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;'>Navigation</div>",
            unsafe_allow_html=True
        )

        nav_options = [
            "Overview",
            "Live Analysis",
            "Analysis History",
            "Model Performance",
            "About the System"
        ]

        # Find current active index
        current_nav = st.session_state["active_nav"]
        if current_nav not in nav_options:
            current_nav = "Overview"

        selected_nav = st.radio(
            "Main Navigation",
            options=nav_options,
            index=nav_options.index(current_nav),
            label_visibility="collapsed",
            key="nav_radio"
        )
        st.session_state["active_nav"] = selected_nav

        st.markdown("<hr style='margin: 18px 0; border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

        # Live Session Counters in Sidebar
        history = st.session_state["history"]
        total_tests = len(history)
        normal_tests = sum(1 for h in history if not h.get("is_anomaly", False))
        anomaly_tests = sum(1 for h in history if h.get("is_anomaly", False))

        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px solid #182236; border-radius:6px; padding:12px; margin-bottom:14px;">
                <div style="font-size:0.7rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Session Diagnostics</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:3px; font-size:0.78rem;">
                    <span style="color:#94A3B8;">Analyses Run:</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:600; color:#F8FAFC;">{total_tests}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:3px; font-size:0.78rem;">
                    <span style="color:#94A3B8;">Normal Confirmed:</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:600; color:#34D399;">{normal_tests}</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.78rem;">
                    <span style="color:#94A3B8;">Anomalous Alerts:</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:600; color:{'#F87171' if anomaly_tests > 0 else '#94A3B8'};">{anomaly_tests}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # System Status Indicator
        dev_str = str(DEVICE).upper()
        is_cuda = "CUDA" in dev_str
        st.markdown(
            f"""
            <div style="font-size:0.72rem; color:#64748B; font-family:'JetBrains Mono', monospace; line-height:1.5;">
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                    <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#10B981; box-shadow:0 0 6px #10B981;"></span>
                    <span style="color:#34D399; font-weight:600;">ALL SYSTEMS OPERATIONAL</span>
                </div>
                <div>HARDWARE: <span style="color:#F8FAFC;">{dev_str}</span></div>
                <div>PIPELINE: <span style="color:#94A3B8;">PyTorch MFCC-40 AE</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================
    # TOP CONTEXT RIBBON
    # ==========================
    render_industrial_header(
        active_section=st.session_state["selected_section"],
        device=str(DEVICE)
    )

    # ==========================
    # ROUTE VIEWS
    # ==========================
    active_nav = st.session_state["active_nav"]

    if active_nav == "Overview":
        render_overview(
            history=st.session_state["history"],
            current_section=st.session_state["selected_section"],
            on_navigate_to_analysis=lambda: st.session_state.update({"active_nav": "Live Analysis"})
        )
    elif active_nav == "Live Analysis":
        render_audio_analysis(
            current_section=st.session_state["selected_section"],
            update_section_callback=update_section
        )
    elif active_nav == "Analysis History":
        render_analysis_history(
            history=st.session_state["history"],
            on_navigate_to_analysis=lambda: st.session_state.update({"active_nav": "Live Analysis"})
        )
    elif active_nav == "Model Performance":
        render_research_performance()
    elif active_nav == "About the System":
        render_about()


if __name__ == "__main__":
    main()
