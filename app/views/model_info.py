"""
AURALIS Model Architecture & Methodology View.
Details the section-specific MFCC Convolutional Autoencoder architecture,
40 MFCC DSP coefficients, unsupervised MSE reconstruction loss formulation,
and empirical 95th-percentile validation threshold calibration.
"""

import streamlit as st
from components.cards import render_scientific_callout
from components.icons import get_svg_icon
from inference import THRESHOLDS, DEVICE


def render_model_info():
    """
    Renders the Model Architecture & Methodology documentation.
    """
    st.markdown("### Model Architecture & Acoustic Pipeline")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:20px; font-size:0.88rem;'>"
        "Technical specification of the Section-Specific MFCC Convolutional Autoencoder, "
        "acoustic feature representation, unsupervised reconstruction scoring, and validation threshold calibration."
        "</p>",
        unsafe_allow_html=True
    )

    # 1. Section-Specific Modeling
    st.markdown("#### 1. Section-Specific Autoencoder Architecture")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; margin-bottom:18px;">
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
                <strong>Instance-Specific Acoustic Baseline Rationale:</strong><br/>
                In industrial manufacturing environments, rotating machines exhibit distinct acoustic signatures caused by 
                manufacturing tolerances, mounting damping, motor stator harmonics, and physical casing geometry. 
                Pooling disparate physical sections into a single monolithic network dilutes machine-specific baselines. 
                AURALIS deploys a dedicated, independently trained Convolutional Autoencoder for each individual physical section 
                (<code>section_00</code>, <code>section_01</code>, and <code>section_02</code>).
            </p>
            <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:14px; font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:#06B6D4; line-height:1.6;">
                <strong>Deep Convolutional Autoencoder Specification:</strong><br/>
                &bull; <strong>Input Tensor:</strong> [Batch, 1, N_MFCC=40, Time_Frames=626] (normalized)<br/>
                &bull; <strong>Encoder:</strong><br/>
                &nbsp;&nbsp;&bull; Conv2d(1 &rarr; 16, kernel=3, stride=2, padding=1) + BatchNorm2d(16) + ReLU<br/>
                &nbsp;&nbsp;&bull; Conv2d(16 &rarr; 32, kernel=3, stride=2, padding=1) + BatchNorm2d(32) + ReLU<br/>
                &nbsp;&nbsp;&bull; Conv2d(32 &rarr; 64, kernel=3, stride=2, padding=1) + BatchNorm2d(64) + ReLU<br/>
                &bull; <strong>Bottleneck Latent Manifold:</strong> Compact operational representation.<br/>
                &bull; <strong>Decoder:</strong><br/>
                &nbsp;&nbsp;&bull; ConvTranspose2d(64 &rarr; 32, kernel=3, stride=2, padding=1, out_pad=1) + BatchNorm2d(32) + ReLU<br/>
                &nbsp;&nbsp;&bull; ConvTranspose2d(32 &rarr; 16, kernel=3, stride=2, padding=1, out_pad=1) + BatchNorm2d(16) + ReLU<br/>
                &nbsp;&nbsp;&bull; ConvTranspose2d(16 &rarr; 1, kernel=3, stride=2, padding=1, out_pad=1)<br/>
                &bull; <strong>Output Cropping:</strong> Formatted to match original input dimension <code>[B, 1, 40, T]</code>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. 40 MFCC Coefficients
    st.markdown("#### 2. Acoustic Feature Extraction: 40 MFCC Coefficients")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; margin-bottom:18px;">
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
                Industrial fan acoustics are composed of continuous broadband turbulent flow superimposed with tonal blade-pass frequencies (BPF) 
                and motor rotation harmonics. Mel-Frequency Cepstral Coefficients (MFCCs) decorrelate spectral filterbank energies using 
                the Discrete Cosine Transform (DCT), providing a compact representation of the spectral envelope:
            </p>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:10px; margin-top:10px;">
                <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px;">
                    <div style="color:#64748B; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Sampling Rate</div>
                    <div style="color:#F8FAFC; font-family:'JetBrains Mono'; font-size:1.05rem; font-weight:700;">16,000 Hz</div>
                    <div style="color:#64748B; font-size:0.72rem;">Audio resampled to 16 kHz</div>
                </div>
                <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px;">
                    <div style="color:#64748B; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Window Size (N_FFT)</div>
                    <div style="color:#F8FAFC; font-family:'JetBrains Mono'; font-size:1.05rem; font-weight:700;">1024 samples</div>
                    <div style="color:#64748B; font-size:0.72rem;">64 ms Hann analysis window</div>
                </div>
                <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px;">
                    <div style="color:#64748B; font-size:0.72rem; font-weight:600; text-transform:uppercase;">Hop Length</div>
                    <div style="color:#F8FAFC; font-family:'JetBrains Mono'; font-size:1.05rem; font-weight:700;">256 samples</div>
                    <div style="color:#64748B; font-size:0.72rem;">16 ms hop (75% overlap)</div>
                </div>
                <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px;">
                    <div style="color:#64748B; font-size:0.72rem; font-weight:600; text-transform:uppercase;">MFCC Bins</div>
                    <div style="color:#F8FAFC; font-family:'JetBrains Mono'; font-size:1.05rem; font-weight:700;">40 coefficients</div>
                    <div style="color:#64748B; font-size:0.72rem;">Captures acoustic harmonics</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Reconstruction Error as Anomaly Score
    st.markdown("#### 3. Unsupervised Reconstruction Error Scoring")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; margin-bottom:18px;">
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
                Because physical machine anomalies are rare and diverse in real factories, the model is trained 
                <strong>exclusively on healthy operational recordings</strong>. When presented with anomalous acoustics 
                (bearing rattle, blade friction, foreign object ingestion), the autoencoder fails to reconstruct the out-of-distribution patterns:
            </p>
            <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px 16px; margin-bottom:12px;">
                <div style="font-family:'JetBrains Mono'; font-size:0.95rem; color:#06B6D4; font-weight:700;">
                    MSE Reconstruction Loss = (1 / N) &sum; (x<sub>i</sub> - x̂<sub>i</sub>)&sup2;
                </div>
                <div style="color:#64748B; font-size:0.75rem; margin-top:4px;">
                    Where <code>x</code> is the normalized 40-bin MFCC tensor, <code>x̂</code> is the autoencoder's output reconstruction, 
                    and <code>N</code> is the total number of tensor elements (Bins &times; Time Frames).
                </div>
            </div>
            <p style="color:#94A3B8; font-size:0.82rem; line-height:1.5; margin:0;">
                &bull; <strong>Normal fan sound:</strong> Low reconstruction error (autoencoder accurately reproduces learned healthy manifold).<br/>
                &bull; <strong>Degraded fan sound:</strong> High reconstruction error (autoencoder cannot reconstruct abnormal harmonic components).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4. 95th-Percentile Threshold Calibration
    st.markdown("#### 4. 95th-Percentile Normal-Validation Calibration")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; margin-bottom:18px;">
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
                In industrial manufacturing, minimizing false alarms is a critical requirement to avoid unneeded production interruptions. 
                Thresholds are calibrated <strong>strictly on the normal validation split</strong> (101 normal recordings per section, seeded 90/10 split) 
                at the empirical <strong>95th percentile (P95)</strong>, targeting an expected ~5% false alarm rate on healthy machines:
            </p>
            <table style="width:100%; border-collapse:collapse; font-size:0.82rem; font-family:'JetBrains Mono', monospace; background:#080C14; border:1px solid #182236; border-radius:6px; overflow:hidden;">
                <thead>
                    <tr style="background:#131929; color:#06B6D4; text-align:left; border-bottom:1px solid #182236;">
                        <th style="padding:8px 12px;">Section ID</th>
                        <th style="padding:8px 12px;">Locked P95 Threshold</th>
                        <th style="padding:8px 12px;">Normal Validation N</th>
                        <th style="padding:8px 12px;">Observed Validation FPR</th>
                        <th style="padding:8px 12px;">Decision Policy</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #182236;">
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_00</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00347179</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 recordings</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                        <td style="padding:8px 12px; color:#CBD5E1;">Score &gt; Threshold &rarr; Anomaly</td>
                    </tr>
                    <tr style="border-bottom:1px solid #182236;">
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_01</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00448493</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 recordings</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                        <td style="padding:8px 12px; color:#CBD5E1;">Score &gt; Threshold &rarr; Anomaly</td>
                    </tr>
                    <tr>
                        <td style="padding:8px 12px; color:#F8FAFC;"><strong>section_02</strong></td>
                        <td style="padding:8px 12px; color:#F59E0B;">0.00319858</td>
                        <td style="padding:8px 12px; color:#94A3B8;">101 recordings</td>
                        <td style="padding:8px 12px; color:#10B981;">5/101 (4.95%)</td>
                        <td style="padding:8px 12px; color:#CBD5E1;">Score &gt; Threshold &rarr; Anomaly</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. PyTorch Inference Pipeline
    st.markdown("#### 5. Real-Time PyTorch Inference Pipeline")
    st.markdown(
        f"""
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px 20px; margin-bottom:18px;">
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
                The execution pipeline in <code>app/inference.py</code> loads the frozen PyTorch checkpoint state dictionaries 
                and evaluates with zero gradient overhead:
            </p>
            <div style="background:#080C14; border:1px solid #182236; border-radius:6px; padding:12px 14px; font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:#CBD5E1; line-height:1.5;">
                1. <strong>Audio Ingestion:</strong> <code>soundfile.read()</code> &rarr; Mono average &rarr; Resample to 16 kHz<br/>
                2. <strong>MFCC Extraction:</strong> <code>librosa.feature.mfcc(sr=16000, n_mfcc=40, n_fft=1024, hop=256)</code><br/>
                3. <strong>Checkpoint Standardization:</strong> <code>(MFCC - Mean<sub>section</sub>) / (Std<sub>section</sub> + 1e-8)</code><br/>
                4. <strong>Hardware Acceleration:</strong> Tensor evaluated on <strong>{str(DEVICE).upper()}</strong><br/>
                5. <strong>Reconstruction:</strong> <code>with torch.no_grad(): score = torch.mean((model(x) - x) ** 2).item()</code><br/>
                6. <strong>Telemetry Dispatch:</strong> Returns score, threshold, ratio, and binary verdict.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Scientific Notice Callout
    render_scientific_callout()
