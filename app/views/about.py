"""
AURALIS About the System View.
Documents project provenance, MIMII DUE fan acoustic dataset attribution,
CC BY-NC-SA 4.0 licensing, and non-contact acoustic condition monitoring principles.
"""

import streamlit as st
from components.icons import get_svg_icon


def render_about():
    """
    Renders the About the System page.
    """
    st.markdown("### About AURALIS Industrial Intelligence")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:20px; font-size:0.88rem;'>"
        "Industrial IoT condition monitoring framework, dataset provenance, and non-contact acoustic surveillance methodology."
        "</p>",
        unsafe_allow_html=True
    )

    # Architectural Overview
    st.markdown("#### Operational Objective & Problem Formulation")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; font-size:0.88rem; color:#CBD5E1; line-height:1.65; margin-bottom:20px;">
            <p>
                In high-uptime manufacturing facilities and continuous process plants, rotating equipment such as industrial ventilation fans, 
                exhaust blowers, and motor-driven cooling arrays represent critical single points of failure. Unplanned mechanical breakdowns 
                lead to severe production downtime, secondary motor stator burnouts, and safety hazards.
            </p>
            <p>
                <strong>Non-Contact Acoustic Condition Monitoring:</strong> While traditional predictive maintenance relies on intrusive 
                vibration accelerometers mounted directly onto bearing housings, microphone-based acoustic monitoring offers 
                <strong>continuous non-invasive surveillance</strong> that can be deployed rapidly without interrupting machine operation.
            </p>
            <p style="margin-bottom:0;">
                <strong>Unsupervised Anomaly Formulation:</strong> In real factory environments, mechanical failure recordings are rare, 
                heterogeneous, and hazardous to simulate. Consequently, supervised classifiers are impractical. 
                AURALIS deploys an <strong>unsupervised Convolutional Autoencoder</strong> trained exclusively on healthy operational sound 
                to detect out-of-distribution acoustic shifts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dataset Attribution & Technical Specs
    col_ds1, col_ds2 = st.columns([1, 1])

    with col_ds1:
        st.markdown("#### MIMII DUE Dataset Attribution")
        st.markdown(
            """
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px; font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                <div style="font-weight:600; color:#06B6D4; font-size:0.92rem; margin-bottom:6px;">
                    MIMII DUE (Industrial Fan Acoustic Subset)
                </div>
                <p>
                    <strong>Full Title:</strong> Malfunctioning Industrial Machine Investigation and Examination under Domain Shifted Conditions.<br/>
                    <strong>Zenodo Record:</strong> <a href="https://doi.org/10.5281/zenodo.4740355" target="_blank" style="color:#06B6D4; text-decoration:none;">10.5281/zenodo.4740355</a><br/>
                    <strong>License:</strong> <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" style="color:#06B6D4; text-decoration:none;">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)</a><br/>
                    <strong>Target Asset:</strong> Industrial ventilation and cooling fans across three distinct physical sections.
                </p>
                <p style="margin-bottom:0; color:#94A3B8; font-size:0.8rem;">
                    Recordings were acquired via an 8-channel microphone array placed in proximity to operating machinery in real factory acoustics, 
                    resampled to 16 kHz mono for deep autoencoder feature extraction.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ds2:
        st.markdown("#### Dataset Split Hierarchy & Partitions")
        st.markdown(
            """
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px; font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                <table style="width:100%; border-collapse:collapse; font-size:0.8rem; font-family:'JetBrains Mono', monospace;">
                    <thead>
                        <tr style="color:#06B6D4; border-bottom:1px solid #1E2638; text-align:left;">
                            <th style="padding:6px 0;">Split Partition</th>
                            <th style="padding:6px 0;">Count</th>
                            <th style="padding:6px 0;">Composition</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px 0; color:#F8FAFC;">Training Pool</td>
                            <td style="padding:6px 0; color:#94A3B8;">3,009</td>
                            <td style="padding:6px 0; color:#10B981;">1,003 Normal / Section</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px 0; color:#F8FAFC;">Normal Validation</td>
                            <td style="padding:6px 0; color:#94A3B8;">303 (101/sec)</td>
                            <td style="padding:6px 0; color:#F59E0B;">10% Seeded Split (P95)</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:6px 0; color:#F8FAFC;">Source Test Split</td>
                            <td style="padding:6px 0; color:#94A3B8;">600</td>
                            <td style="padding:6px 0; color:#94A3B8;">100 Norm + 100 Anom / sec</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#F8FAFC;">Target Test Split</td>
                            <td style="padding:6px 0; color:#94A3B8;">600</td>
                            <td style="padding:6px 0; color:#EF4444;">Domain Shifted (100+100/sec)</td>
                        </tr>
                    </tbody>
                </table>
                <div style="margin-top:10px; font-size:0.75rem; color:#64748B;">
                    Sections: <code>section_00</code>, <code>section_01</code>, and <code>section_02</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown("#### System Technology Stack")
    st.markdown(
        """
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:10px;">
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:12px;">
                <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Deep Learning</div>
                <div style="color:#F8FAFC; font-size:1rem; font-weight:600; margin:4px 0;">PyTorch 2.14</div>
                <div style="color:#64748B; font-size:0.72rem;">CUDA GPU accelerated tensor math</div>
            </div>
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:12px;">
                <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Digital Signal Processing</div>
                <div style="color:#F8FAFC; font-size:1rem; font-weight:600; margin:4px 0;">Librosa & SoundFile</div>
                <div style="color:#64748B; font-size:0.72rem;">40 MFCC extraction & filtering</div>
            </div>
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:12px;">
                <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Frontend & Visualization</div>
                <div style="color:#F8FAFC; font-size:1rem; font-weight:600; margin:4px 0;">Streamlit & Altair</div>
                <div style="color:#64748B; font-size:0.72rem;">Dark industrial SCADA interface</div>
            </div>
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:12px;">
                <div style="font-size:0.72rem; color:#64748B; font-weight:600; text-transform:uppercase;">Neural Topology</div>
                <div style="color:#F8FAFC; font-size:1rem; font-weight:600; margin:4px 0;">CNN Autoencoder</div>
                <div style="color:#64748B; font-size:0.72rem;">Unit-specific latent manifolds</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
