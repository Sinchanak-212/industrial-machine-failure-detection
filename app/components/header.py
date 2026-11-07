"""
AURALIS Top Navigation & Telemetry Context Bar.
Provides enterprise SaaS contextual metadata inspired by real industrial IoT consoles.
"""

import streamlit as st
from components.icons import get_svg_icon


def render_industrial_header(active_section: str = "section_00", device: str = "cuda"):
    """
    Renders the AURALIS enterprise context bar.
    """
    dev_str = str(device).upper()
    is_cuda = "CUDA" in dev_str

    st.markdown(
        f"""
        <div class="context-ribbon">
            <div class="context-chips-group">
                <div class="context-chip">
                    {get_svg_icon('server', 13, '#94A3B8')}
                    <span>PLANT:</span>
                    <strong>North Fan Array</strong>
                </div>
                <div class="context-chip active-cyan">
                    {get_svg_icon('activity', 13, '#06B6D4')}
                    <span>TARGET SECTION:</span>
                    <strong>{active_section.upper()}</strong>
                </div>
                <div class="context-chip">
                    {get_svg_icon('shield', 13, '#94A3B8')}
                    <span>MODEL:</span>
                    <strong>MFCC CNN-AE (Frozen)</strong>
                </div>
            </div>
            
            <div class="context-chips-group">
                <div class="context-chip active-green">
                    <span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:#10B981; box-shadow:0 0 6px #10B981;"></span>
                    <span>SYSTEM:</span>
                    <strong>OPERATIONAL</strong>
                </div>
                <div class="context-chip">
                    {get_svg_icon('cpu', 13, '#38BDF8' if is_cuda else '#F59E0B')}
                    <span>BACKEND:</span>
                    <strong>{dev_str}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
