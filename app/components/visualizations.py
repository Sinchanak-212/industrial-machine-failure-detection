"""
AURALIS Acoustic & Score Visualization Engine.
Provides industrial telemetry visualizations:
1. Acoustic Waveform (time domain, 16 kHz)
2. 40 MFCC Spectral Coefficient Heatmap (dB scale)
3. Industrial Score vs. Threshold Gauge & Bullet Graph
"""

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def create_waveform_and_mfcc_plot(audio: np.ndarray, sr: int, mfccs: np.ndarray):
    """
    Renders an industrial dark-mode plot containing the acoustic waveform
    and the 40 MFCC coefficient heat map.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 4.4),
        facecolor="#0E131F",
        gridspec_kw={"height_ratios": [1, 1.2]}
    )
    
    # 1. Waveform Plot
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax1.set_facecolor("#080C14")
    ax1.plot(time_axis, audio, color="#06B6D4", linewidth=0.7, alpha=0.95)
    ax1.fill_between(time_axis, audio, color="#06B6D4", alpha=0.12)
    ax1.set_xlim(0, max(time_axis[-1], 0.1))
    ax1.set_ylabel("Amplitude", color="#94A3B8", fontsize=8, fontweight="bold")
    ax1.set_title("Acoustic Waveform (Time Domain @ 16 kHz)", color="#F8FAFC", fontsize=9, fontweight="bold", loc="left", pad=6)
    ax1.tick_params(colors="#64748B", labelsize=8)
    ax1.grid(True, linestyle="--", alpha=0.12, color="#94A3B8")
    for spine in ax1.spines.values():
        spine.set_color("#182236")

    # 2. MFCC Coefficients Heatmap
    ax2.set_facecolor("#080C14")
    im = ax2.imshow(
        mfccs,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest"
    )
    ax2.set_ylabel("MFCC Bins (0-39)", color="#94A3B8", fontsize=8, fontweight="bold")
    ax2.set_xlabel("Time Frames (Hop=256, N_FFT=1024)", color="#94A3B8", fontsize=8, fontweight="bold")
    ax2.set_title("40 MFCC Coefficients (Spectral Envelope)", color="#F8FAFC", fontsize=9, fontweight="bold", loc="left", pad=6)
    ax2.tick_params(colors="#64748B", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#182236")
        
    cbar = fig.colorbar(im, ax=ax2, pad=0.015, aspect=14)
    cbar.set_label("dB Scale", color="#94A3B8", fontsize=8)
    cbar.ax.tick_params(colors="#64748B", labelsize=8)
    cbar.outline.set_edgecolor("#182236")

    plt.tight_layout()
    return fig


def render_score_vs_threshold_chart(anomaly_score: float, threshold: float, section: str):
    """
    Renders an enterprise Score vs. Threshold telemetry visualizer.
    """
    is_anomaly = anomaly_score > threshold
    ratio = (anomaly_score / threshold) if threshold > 0 else 1.0

    max_val = max(anomaly_score, threshold) * 1.35
    score_color = "#EF4444" if is_anomaly else "#06B6D4"
    status_label = "ANOMALOUS (> Threshold)" if is_anomaly else "NORMAL (<= Threshold)"

    df = pd.DataFrame([
        {
            "Metric": "Validation Threshold (P95)",
            "Score": threshold,
            "Type": "Threshold",
            "Color": "#F59E0B"
        },
        {
            "Metric": "Reconstruction Error",
            "Score": anomaly_score,
            "Type": "Reconstruction Error",
            "Color": score_color
        }
    ])

    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=24).encode(
        x=alt.X(
            "Score:Q",
            title="Autoencoder Reconstruction Error (MSE)",
            scale=alt.Scale(domain=[0, max_val]),
            axis=alt.Axis(
                grid=True,
                gridColor="#182236",
                labelColor="#94A3B8",
                titleColor="#94A3B8",
                format=".6f"
            )
        ),
        y=alt.Y(
            "Metric:N",
            title=None,
            axis=alt.Axis(
                labelColor="#F8FAFC",
                labelFontSize=11,
                labelFontWeight="bold"
            )
        ),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(
                domain=["Threshold", "Reconstruction Error"],
                range=["#F59E0B", score_color]
            ),
            legend=alt.Legend(
                title=None,
                orient="top",
                labelColor="#94A3B8"
            )
        ),
        tooltip=[
            alt.Tooltip("Metric:N", title="Telemetry Point"),
            alt.Tooltip("Score:Q", title="Value", format=".6f")
        ]
    ).properties(
        height=130,
        background="#0E131F"
    ).configure_view(
        strokeWidth=0
    )

    st.altair_chart(chart, use_container_width=True)

    # Clean industrial progress meter
    fill_width = min(max(int((anomaly_score / max_val) * 100), 2), 100)
    thresh_pos = min(max(int((threshold / max_val) * 100), 2), 98)

    st.markdown(
        f"""
        <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px 16px; margin-top:6px;">
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#64748B; margin-bottom:6px; font-family:'JetBrains Mono', monospace;">
                <span>BASELINE: 0.000000</span>
                <span style="color:#F59E0B; font-weight:600;">CALIBRATED THRESHOLD: {threshold:.6f}</span>
                <span>MAX SCALE: {max_val:.6f}</span>
            </div>
            <div style="position:relative; height:14px; background:#111622; border-radius:4px; overflow:hidden;">
                <!-- Threshold Marker line -->
                <div style="position:absolute; left:{thresh_pos}%; top:0; bottom:0; width:2px; background:#F59E0B; z-index:10; box-shadow:0 0 4px #F59E0B;"></div>
                <!-- Score fill -->
                <div style="height:100%; width:{fill_width}%; background:{score_color}; border-radius:4px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; font-size:0.78rem;">
                <span style="color:#94A3B8;">Machine: <strong style="color:#F8FAFC;">{section}</strong></span>
                <span style="font-family:'JetBrains Mono', monospace; color:{score_color}; font-weight:600;">
                    {status_label} &bull; Score: {anomaly_score:.6f} ({ratio:.2f}x)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
