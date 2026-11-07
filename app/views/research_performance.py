"""
AURALIS Model Performance View.
Displays empirical benchmark metrics, bootstrap confidence intervals,
threshold sensitivity analysis, and explicit scientific disclosures.
Matches verified reports from reports/FINAL_EXPERIMENT_ANALYSIS.md and reports/CONFIDENCE_INTERVAL_ANALYSIS.md.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from components.cards import render_telemetry_card, render_scientific_callout
from components.icons import get_svg_icon


def render_research_performance():
    """
    Renders the Model Performance & Benchmark Evaluation page.
    """
    st.markdown("### Model Benchmark Performance & Evaluation")
    st.markdown(
        "<p style='color:#94A3B8; margin-top:-8px; margin-bottom:18px; font-size:0.88rem;'>"
        "Empirical benchmark results on the MIMII DUE fan dataset. All metrics derived from frozen checkpoints "
        "and locked 95th-percentile normal validation thresholds without test-domain tuning."
        "</p>",
        unsafe_allow_html=True
    )

    # Explicit System Scope & Sensitivity Limitation Notice
    st.markdown(
        f"""
        <div style="background:rgba(239, 68, 68, 0.06); border-left:3px solid #EF4444; border-radius:0 6px 6px 0; padding:16px 20px; margin-bottom:20px;">
            <div style="font-weight:700; color:#F87171; font-size:0.95rem; display:flex; align-items:center; gap:8px;">
                <span>{get_svg_icon('alert_triangle', 18, '#EF4444')}</span>
                <span>SYSTEM PERFORMANCE BOUNDARY & SCIENTIFIC DISCLOSURE</span>
            </div>
            <div style="color:#CBD5E1; font-size:0.86rem; line-height:1.6; margin-top:6px;">
                The model possesses <b>limited anomaly detection sensitivity (5.67% target macro recall)</b> at the validation-calibrated 
                95th-percentile threshold. It serves as an <b>experimental acoustic anomaly detection prototype</b>, 
                <b>not a validated predictive maintenance solution</b>. It provides unsupervised detection of generalized distribution shifts, 
                but does not classify failure modes, estimate remaining useful life (RUL), or predict impending mechanical breakdown.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5 Verified Core Metrics KPI Grid
    st.markdown("#### Target Test Split: Macro-Averaged Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_telemetry_card(
            label="Macro ROC-AUC",
            value="0.6091",
            subtext="95% CI: [0.5651, 0.6543]",
            icon_name="performance",
            accent="cyan"
        )
    with col2:
        render_telemetry_card(
            label="Macro Recall",
            value="5.67%",
            subtext="95% CI: [3.33%, 8.33%]",
            icon_name="alert_triangle",
            accent="red"
        )
    with col3:
        render_telemetry_card(
            label="Macro Precision",
            value="49.81%",
            subtext="95% CI: [32.26%, 67.25%]",
            icon_name="check_circle",
            accent="amber"
        )
    with col4:
        render_telemetry_card(
            label="Macro F1-Score",
            value="10.16%",
            subtext="95% CI: [5.83%, 14.48%]",
            icon_name="activity",
            accent="default"
        )
    with col5:
        render_telemetry_card(
            label="Macro Accuracy",
            value="50.17%",
            subtext="95% CI: [48.33%, 51.83%]",
            icon_name="shield",
            accent="default"
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Comparative Model Benchmark Table
    st.markdown("#### Benchmark Comparison Across All Evaluated Architectures")
    st.markdown(
        "<p style='color:#64748B; font-size:0.8rem; margin-top:-6px;'>"
        "All models evaluated under identical frozen validation-threshold conditions. Model selection locked on Source Test macro ROC-AUC (leader: Section MFCC CNN = 0.5698)."
        "</p>",
        unsafe_allow_html=True
    )

    models_data = [
        {"Model Architecture": "Section-specific MFCC CNN (Selected)", "Test Set": "Source Test", "ROC-AUC": 0.5698, "Accuracy": "51.17%", "Precision": "58.68%", "Recall": "6.67%", "F1": "11.75%"},
        {"Model Architecture": "MFCC frame-level P95", "Test Set": "Source Test", "ROC-AUC": 0.5668, "Accuracy": "51.17%", "Precision": "54.32%", "Recall": "9.67%", "F1": "15.90%"},
        {"Model Architecture": "MFCC + Isolation Forest", "Test Set": "Source Test", "ROC-AUC": 0.5616, "Accuracy": "52.33%", "Precision": "65.13%", "Recall": "9.33%", "F1": "15.97%"},
        {"Model Architecture": "Section-specific Mel CNN", "Test Set": "Source Test", "ROC-AUC": 0.5545, "Accuracy": "52.00%", "Precision": "57.42%", "Recall": "12.67%", "F1": "20.69%"},
        {"Model Architecture": "Shared Mel CNN", "Test Set": "Source Test", "ROC-AUC": 0.5348, "Accuracy": "51.33%", "Precision": "68.52%", "Recall": "9.33%", "F1": "15.50%"},
        {"Model Architecture": "Section-specific MFCC CNN (Selected)", "Test Set": "Target Test", "ROC-AUC": 0.6091, "Accuracy": "50.17%", "Precision": "49.81%", "Recall": "5.67%", "F1": "10.16%"},
        {"Model Architecture": "MFCC + Isolation Forest", "Test Set": "Target Test", "ROC-AUC": 0.6105, "Accuracy": "51.17%", "Precision": "37.65%", "Recall": "8.33%", "F1": "13.52%"},
        {"Model Architecture": "MFCC frame-level P95", "Test Set": "Target Test", "ROC-AUC": 0.5779, "Accuracy": "51.33%", "Precision": "56.25%", "Recall": "9.00%", "F1": "15.41%"},
        {"Model Architecture": "Section-specific Mel CNN", "Test Set": "Target Test", "ROC-AUC": 0.5514, "Accuracy": "52.00%", "Precision": "64.29%", "Recall": "12.00%", "F1": "18.96%"},
        {"Model Architecture": "Shared Mel CNN", "Test Set": "Target Test", "ROC-AUC": 0.5217, "Accuracy": "49.33%", "Precision": "62.84%", "Recall": "7.33%", "F1": "11.91%"}
    ]
    st.dataframe(pd.DataFrame(models_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Per-Section Breakdown Table
    col_sec1, col_sec2 = st.columns([1, 1])
    with col_sec1:
        st.markdown("#### Per-Section Performance (Target Test Split)")
        per_sec_data = [
            {"Section": "section_00", "ROC-AUC": 0.7126, "95% CI": "[0.6361, 0.7808]", "Accuracy": "51.50%", "Precision": "61.54%", "Recall": "8.00%", "F1": "14.16%"},
            {"Section": "section_01", "ROC-AUC": 0.5323, "95% CI": "[0.4519, 0.6108]", "Accuracy": "48.50%", "Precision": "33.33%", "Recall": "3.00%", "F1": "5.50%"},
            {"Section": "section_02", "ROC-AUC": 0.5824, "95% CI": "[0.5064, 0.6602]", "Accuracy": "50.50%", "Precision": "54.55%", "Recall": "6.00%", "F1": "10.81%"},
            {"Section": "Macro Mean", "ROC-AUC": 0.6091, "95% CI": "[0.5651, 0.6543]", "Accuracy": "50.17%", "Precision": "49.81%", "Recall": "5.67%", "F1": "10.16%"}
        ]
        st.dataframe(pd.DataFrame(per_sec_data), use_container_width=True, hide_index=True)

    with col_sec2:
        st.markdown("#### Paired Bootstrap AUC Difference")
        st.markdown(
            """
            <div style="background:#0E131F; border:1px solid #1E2638; border-radius:6px; padding:14px; font-size:0.84rem; color:#CBD5E1; line-height:1.6;">
                <div style="font-weight:600; color:#06B6D4; margin-bottom:4px;">
                    Paired Comparison: Section MFCC CNN minus MFCC + Isolation Forest
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:0.88rem; margin-bottom:6px;">
                    &bull; <strong>Point Difference:</strong> <span style="color:#F59E0B;">-0.0014</span><br/>
                    &bull; <strong>95% Bootstrap CI:</strong> <span style="color:#F59E0B;">[-0.0365, +0.0338]</span>
                </div>
                <p style="color:#94A3B8; font-size:0.8rem; margin:0;">
                    Because the 95% paired bootstrap interval includes zero, the data do <b>not</b> establish a statistically significant difference 
                    between the MFCC CNN and the Isolation Forest. Point estimate ordering alone does not establish model superiority.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Threshold Sensitivity Operating Points
    st.markdown("#### Validation Threshold Sensitivity Trade-offs")
    st.markdown(
        "<p style='color:#64748B; font-size:0.8rem; margin-top:-6px;'>"
        "Empirical operating points across validation percentiles for Section-specific MFCC CNN on 600 target test recordings."
        "</p>",
        unsafe_allow_html=True
    )

    sens_data = [
        {"Percentile": "90.0% (P90)", "Threshold Policy": "Nominal 10% FPR", "Target Recall": "32.00%", "Target Precision": "66.04%", "Target F1": "42.95%", "Target FPR": "17.33%", "FP": 52, "FN": 204, "TP": 96, "TN": 248},
        {"Percentile": "95.0% (P95, Locked)", "Threshold Policy": "Nominal 5% FPR", "Target Recall": "5.67%", "Target Precision": "49.81%", "Target F1": "10.16%", "Target FPR": "5.33%", "FP": 16, "FN": 283, "TP": 17, "TN": 284},
        {"Percentile": "97.5% (P97.5)", "Threshold Policy": "Nominal 2.5% FPR", "Target Recall": "3.00%", "Target Precision": "47.78%", "Target F1": "5.58%", "Target FPR": "3.00%", "FP": 9, "FN": 291, "TP": 9, "TN": 291},
        {"Percentile": "99.0% (P99)", "Threshold Policy": "Nominal 1% FPR", "Target Recall": "1.67%", "Target Precision": "58.33%", "Target F1": "3.23%", "Target FPR": "1.00%", "FP": 3, "FN": 295, "TP": 5, "TN": 297}
    ]
    st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Embedded Research Plots from reports/
    st.markdown("#### Empirical Evaluation Artifacts & Diagnostic Charts")
    
    report_tabs = st.tabs([
        "ROC Curves",
        "Confidence Intervals",
        "Confusion Matrices",
        "Score Distributions",
        "Threshold Sensitivity"
    ])

    reports_dir = Path("reports")

    with report_tabs[0]:
        img_path = reports_dir / "strongest_model_roc_curves.png"
        if img_path.exists():
            st.image(str(img_path), caption="Target Test ROC Curves ranked by Source Test ROC-AUC", use_container_width=True)

    with report_tabs[1]:
        col_ci1, col_ci2 = st.columns([1, 1])
        with col_ci1:
            img_path = reports_dir / "roc_auc_confidence_intervals.png"
            if img_path.exists():
                st.image(str(img_path), caption="95% Bootstrap Confidence Intervals for Macro ROC-AUC", use_container_width=True)
        with col_ci2:
            img_path = reports_dir / "paired_auc_difference.png"
            if img_path.exists():
                st.image(str(img_path), caption="Paired Bootstrap AUC Difference Distribution (MFCC CNN vs IsoForest)", use_container_width=True)

    with report_tabs[2]:
        img_path = reports_dir / "final_selected_confusion_matrices.png"
        if img_path.exists():
            st.image(str(img_path), caption="Target Test Confusion Matrices at Locked P95 Threshold", use_container_width=True)

    with report_tabs[3]:
        col_dist1, col_dist2 = st.columns([1, 1])
        with col_dist1:
            img_path = reports_dir / "anomaly_score_distributions.png"
            if img_path.exists():
                st.image(str(img_path), caption="Target Test Anomaly Score Distributions (Normal vs Anomalous)", use_container_width=True)
        with col_dist2:
            img_path = reports_dir / "validation_score_distributions.png"
            if img_path.exists():
                st.image(str(img_path), caption="Normal Validation Score Distributions & P95 Threshold Cutoffs", use_container_width=True)

    with report_tabs[4]:
        img_path = reports_dir / "threshold_sensitivity.png"
        if img_path.exists():
            st.image(str(img_path), caption="Threshold Sensitivity: Recall, Precision, and FPR Trade-offs", use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Detailed Explanation of Scientific Limitations
    st.markdown("#### Scientific Analysis of Low Sensitivity & Engineering Insights")
    st.markdown(
        """
        <div style="background:#0E131F; border:1px solid #1E2638; border-radius:8px; padding:18px; font-size:0.86rem; color:#CBD5E1; line-height:1.65;">
            <div style="font-weight:600; color:#06B6D4; font-size:0.95rem; margin-bottom:8px;">
                Why is Target Macro Recall 5.67% at the 95th-Percentile Operating Point?
            </div>
            <p>
                1. <strong>Mean Pooling Dilution Across 10-Second Clips:</strong> The Autoencoder calculates average reconstruction error 
                across the entire 10-second recording. Many physical machinery defects exhibit brief, transient clicks, periodic friction, 
                or intermittent rubbing. Averaging over 626 temporal frames dilutes localized high-error spikes against the predominantly normal baseline noise.
            </p>
            <p>
                2. <strong>Acoustic Domain Shift (Source vs. Target):</strong> The MIMII DUE dataset benchmarks domain adaptation. 
                Target test recordings introduce shifts in ambient factory background noise, operational motor load, and microphone distance. 
                Because the autoencoder was trained strictly on normal recordings from the source domain, healthy sounds in the target domain 
                experience subtle distribution shifts that overlap with minor anomalies.
            </p>
            <p>
                3. <strong>Normal-Only Validation Constraint:</strong> Calibrating the threshold at the 95th percentile strictly guarantees 
                that ~95% of healthy normal recordings are classified as safe (FPR &approx; 4.95%), strictly enforcing low false alarm frequency. 
                However, because anomaly sound signatures were completely unseen during training, this conservative boundary does not guarantee separation of anomalous scores.
            </p>
            <p style="margin-bottom:0;">
                <strong>Industrial Recommendation:</strong> For production-grade predictive maintenance, this acoustic detector should be coupled 
                with multi-modal vibration accelerometers, thermal infrared telemetry, and temporal frame-level peak anomaly detection.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
