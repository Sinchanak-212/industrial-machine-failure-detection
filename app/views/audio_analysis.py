"""
AURALIS Live Analysis View.
Provides enterprise audio ingestion, WAV validation, benchmark sample loading,
real-time PyTorch CNN Autoencoder inference via predict_audio(),
waveform and MFCC spectrogram generation, and score vs. threshold telemetry.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from components.cards import (
    render_diagnostic_banner,
    render_telemetry_card,
    render_scientific_callout
)
from components.icons import get_svg_icon
from components.visualizations import (
    create_waveform_and_mfcc_plot,
    render_score_vs_threshold_chart
)
from inference import predict_audio, extract_mfcc, THRESHOLDS, DEVICE


# Verified benchmark test audio samples available in repository
BENCHMARK_SAMPLES = {
    "section_00 - Target Test Normal (0000)": {
        "path": "data/raw/fan/target_test/section_00_target_test_normal_0000.wav",
        "section": "section_00",
        "expected": "Normal",
        "description": "Baseline healthy fan operation sound in target domain."
    },
    "section_00 - Target Test Anomaly (0044)": {
        "path": "data/raw/fan/target_test/section_00_target_test_anomaly_0044.wav",
        "section": "section_00",
        "expected": "Anomalous",
        "description": "High reconstruction error anomaly (score ~0.004530 > threshold 0.003472)."
    },
    "section_00 - Source Test Anomaly (0013)": {
        "path": "data/raw/fan/source_test/section_00_source_test_anomaly_0013.wav",
        "section": "section_00",
        "expected": "Anomalous",
        "description": "Source test split anomaly (score ~0.003743 > threshold 0.003472)."
    },
    "section_01 - Target Test Normal (0000)": {
        "path": "data/raw/fan/target_test/section_01_target_test_normal_0000.wav",
        "section": "section_01",
        "expected": "Normal",
        "description": "Baseline healthy sound for machine unit 01."
    },
    "section_01 - Target Test Anomaly (0029)": {
        "path": "data/raw/fan/target_test/section_01_target_test_anomaly_0029.wav",
        "section": "section_01",
        "expected": "Anomalous",
        "description": "Acoustic anomaly for machine unit 01 (score ~0.004775 > threshold 0.004485)."
    },
    "section_02 - Target Test Normal (0000)": {
        "path": "data/raw/fan/target_test/section_02_target_test_normal_0000.wav",
        "section": "section_02",
        "expected": "Normal",
        "description": "Baseline healthy sound for machine unit 02."
    },
    "section_02 - Target Test Anomaly (0017)": {
        "path": "data/raw/fan/target_test/section_02_target_test_anomaly_0017.wav",
        "section": "section_02",
        "expected": "Anomalous",
        "description": "Acoustic anomaly for machine unit 02 (score ~0.004009 > threshold 0.003199)."
    }
}


def render_audio_analysis(current_section: str, update_section_callback):
    """
    Renders the AURALIS Live Analysis studio.
    """
    st.markdown("### Live Acoustic Diagnostics & Signal Analysis")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:18px; font-size:0.88rem;'>"
        "Ingest raw acoustic signals (.wav) for real-time PyTorch CNN Autoencoder feature extraction and reconstruction scoring."
        "</p>",
        unsafe_allow_html=True
    )

    # Machine Section Selection & Active Threshold Bar
    col_sec1, col_sec2 = st.columns([1, 1])

    with col_sec1:
        sections = list(THRESHOLDS.keys())
        default_idx = sections.index(current_section) if current_section in sections else 0
        selected_section = st.selectbox(
            "Target Machine Unit",
            options=sections,
            index=default_idx,
            help="Select the designated machine section. Each unit uses its own trained model and threshold."
        )
        if selected_section != current_section:
            update_section_callback(selected_section)

    with col_sec2:
        section_thresh = THRESHOLDS[selected_section]
        st.markdown(
            f"""
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:10px 14px; margin-top:24px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.72rem; color:#94A3B8; font-weight:600; text-transform:uppercase;">Unit Cutoff Threshold (P95):</span>
                    <span style="font-family:'JetBrains Mono', monospace; color:#F59E0B; font-weight:700;">{section_thresh:.8f}</span>
                </div>
                <div style="font-size:0.72rem; color:#64748B; margin-top:2px;">
                    Calibrated on 101 normal validation recordings (nom. 5% FPR)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Audio Ingestion Tabs
    tab_upload, tab_bench = st.tabs([
        "WAV File Ingestion",
        "Benchmark Test Corpus"
    ])

    audio_bytes_to_play = None
    target_audio_path = None
    display_filename = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload Machine Sound Recording (WAV format, 16 kHz mono, ~10s duration)",
            type=["wav"],
            key="custom_wav_uploader",
            help="Upload an uncompressed WAV sound recording of the machine fan."
        )
        st.markdown(
            """
            <div style="font-size:0.75rem; color:#64748B; margin-top:4px; font-family:'JetBrains Mono', monospace;">
                SUPPORTED FORMATS: PCM 16-bit WAV &bull; Resampled automatically to 16,000 Hz &bull; Channels averaged to mono
            </div>
            """,
            unsafe_allow_html=True
        )
        if uploaded_file is not None:
            if not uploaded_file.name.lower().endswith(".wav"):
                st.error("Invalid file format. Please upload a standard PCM .WAV audio file.")
            else:
                display_filename = uploaded_file.name
                audio_bytes_to_play = uploaded_file.getvalue()

    with tab_bench:
        st.markdown(
            "<div style='font-size:0.82rem; color:#94A3B8; margin-bottom:8px;'>"
            "Select verified acoustic test recordings from the MIMII DUE fan dataset splits for instant diagnostic verification:"
            "</div>",
            unsafe_allow_html=True
        )
        selected_sample_key = st.selectbox(
            "Verified Benchmark Audio Recordings",
            options=list(BENCHMARK_SAMPLES.keys()),
            key="benchmark_sample_selector"
        )
        
        sample_meta = BENCHMARK_SAMPLES[selected_sample_key]
        sample_rel_path = sample_meta["path"]
        
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#CBD5E1; background:#080C14; border:1px solid #182236; border-radius:6px; padding:10px 14px; margin-bottom:12px;">
                <div><strong>Description:</strong> {sample_meta['description']}</div>
                <div style="margin-top:4px;">
                    <strong>Expected Diagnosis:</strong> <span style="font-family:'JetBrains Mono'; color:{'#F87171' if sample_meta['expected']=='Anomalous' else '#06B6D4'};">{sample_meta['expected']}</span> &bull; 
                    <strong>Designated Section:</strong> <span style="font-family:'JetBrains Mono'; color:#38BDF8;">{sample_meta['section']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            if st.button("Load Benchmark Recording", key="btn_load_benchmark", use_container_width=True):
                st.session_state["active_sample_key"] = selected_sample_key
                if sample_meta["section"] != selected_section:
                    update_section_callback(sample_meta["section"])
                    st.rerun()

        if st.session_state.get("active_sample_key") == selected_sample_key:
            sample_file_path = Path(sample_rel_path)
            if sample_file_path.exists():
                target_audio_path = str(sample_file_path)
                display_filename = sample_file_path.name
                with open(sample_file_path, "rb") as f:
                    audio_bytes_to_play = f.read()
            else:
                st.warning(f"Benchmark file not found at: {sample_rel_path}")

    # Audio Playback and Inference Trigger
    if audio_bytes_to_play is not None and display_filename is not None:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Audio Stream Preview & Execution")
        
        col_audio, col_btn = st.columns([2, 1])
        with col_audio:
            st.audio(audio_bytes_to_play, format="audio/wav")
            st.markdown(
                f"<div style='font-size:0.75rem; color:#64748B; font-family:\"JetBrains Mono\"; margin-top:2px;'>"
                f"Source: <strong>{display_filename}</strong> &bull; Assigned Unit: <strong>{selected_section}</strong>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_btn:
            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
            run_analysis_clicked = st.button(
                "Run Acoustic Inference",
                type="primary",
                use_container_width=True
            )

        # Execution Logic
        if run_analysis_clicked:
            temp_path = None
            try:
                if uploaded_file is not None:
                    temp_dir = tempfile.gettempdir()
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                        dir=temp_dir
                    )
                    temp_file.write(audio_bytes_to_play)
                    temp_file.flush()
                    temp_file.close()
                    temp_path = temp_file.name
                    path_to_infer = temp_path
                else:
                    path_to_infer = target_audio_path

                with st.spinner("Extracting 40 MFCC coefficients & evaluating Autoencoder reconstruction loss..."):
                    start_time = time.perf_counter()
                    
                    # Call real inference function
                    result = predict_audio(path_to_infer, selected_section)
                    
                    # Extract signals for display
                    audio_series, sr = sf.read(path_to_infer)
                    if audio_series.ndim > 1:
                        audio_series = audio_series.mean(axis=1)
                    if sr != 16000:
                        audio_series = librosa.resample(audio_series.astype(np.float32), orig_sr=sr, target_sr=16000)
                        sr = 16000
                    
                    mfcc_feat = extract_mfcc(path_to_infer)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": display_filename,
                    "section": selected_section,
                    "anomaly_score": result["anomaly_score"],
                    "threshold": result["threshold"],
                    "prediction": result["prediction"],
                    "is_anomaly": result["is_anomaly"],
                    "device": result["device"],
                    "latency_ms": latency_ms
                }
                
                st.session_state["history"].insert(0, record)
                st.session_state["last_result"] = {
                    "record": record,
                    "audio": audio_series,
                    "sr": sr,
                    "mfccs": mfcc_feat,
                    "audio_bytes": audio_bytes_to_play
                }

                st.success(f"Inference completed in {latency_ms:.1f} ms on {result['device'].upper()}")

            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass

    # Display Current Diagnostic Results
    if "last_result" in st.session_state and st.session_state["last_result"] is not None:
        last_res = st.session_state["last_result"]
        rec = last_res["record"]

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Diagnostic Telemetry Verdict")

        render_diagnostic_banner(
            prediction=rec["prediction"],
            anomaly_score=rec["anomaly_score"],
            threshold=rec["threshold"],
            filename=rec["filename"],
            device=rec["device"]
        )

        # 4 Quantitative Telemetry Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        delta = rec["anomaly_score"] - rec["threshold"]
        delta_str = f"{delta:+.6f}"
        score_ratio = rec["anomaly_score"] / rec["threshold"] if rec["threshold"] > 0 else 1.0

        with col_m1:
            render_telemetry_card(
                label="Reconstruction Error",
                value=f"{rec['anomaly_score']:.6f}",
                subtext="MSE reconstruction loss",
                icon_name="activity",
                accent="red" if rec["is_anomaly"] else "cyan"
            )
        with col_m2:
            render_telemetry_card(
                label="Validation Threshold",
                value=f"{rec['threshold']:.6f}",
                subtext=f"P95 cutoff for {rec['section']}",
                icon_name="shield",
                accent="amber"
            )
        with col_m3:
            render_telemetry_card(
                label="Threshold Margin / Delta",
                value=delta_str,
                subtext=f"{score_ratio:.2f}x of safety cutoff",
                icon_name="alert_triangle" if rec["is_anomaly"] else "check_circle",
                accent="red" if rec["is_anomaly"] else "cyan"
            )
        with col_m4:
            render_telemetry_card(
                label="Inference Latency",
                value=f"{rec.get('latency_ms', 0):.1f} ms",
                subtext=f"Hardware: {rec['device'].upper()}",
                icon_name="cpu",
                accent="default"
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Anomaly Score vs Threshold Gauge / Bullet
        st.markdown("##### Anomaly Score vs. Detection Threshold")
        render_score_vs_threshold_chart(
            anomaly_score=rec["anomaly_score"],
            threshold=rec["threshold"],
            section=rec["section"]
        )

        st.markdown(
            """
            <div style="background:#0E131F; border:1px solid #182236; border-radius:6px; padding:12px 16px; margin-top:8px; font-size:0.8rem; color:#94A3B8; line-height:1.5;">
                <strong style="color:#F8FAFC;">Diagnostic Interpretation:</strong> The <strong>Reconstruction Error</strong> represents the Mean Squared Error between the original 40 MFCC features and the Autoencoder's reconstructed tensor. Because the model was trained strictly on healthy machine sounds, unfamiliar acoustical vibrations yield higher reconstruction error. If the score exceeds the <strong>Validation Threshold</strong> (calibrated at the 95th percentile of normal operational sounds), an anomaly alert is raised.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Acoustic Signal Inspection
        st.markdown("##### Acoustic Signal Inspection (Waveform & 40 MFCC Heatmap)")
        with st.spinner("Rendering spectral visualizations..."):
            fig = create_waveform_and_mfcc_plot(
                audio=last_res["audio"],
                sr=last_res["sr"],
                mfccs=last_res["mfccs"]
            )
            st.pyplot(fig)

        # Export Current Diagnostic
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns([1, 1])
        with col_exp1:
            json_str = json.dumps(rec, indent=2)
            st.download_button(
                label="Export Diagnostic Report (JSON)",
                data=json_str,
                file_name=f"auralis_diagnostic_{rec['filename']}.json",
                mime="application/json"
            )
        with col_exp2:
            df_curr = pd.DataFrame([rec])
            st.download_button(
                label="Export Diagnostic Record (CSV)",
                data=df_curr.to_csv(index=False).encode("utf-8"),
                file_name=f"auralis_diagnostic_{rec['filename']}.csv",
                mime="text/csv"
            )

    # Scientific Notice Callout
    render_scientific_callout()
