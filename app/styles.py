"""
AURALIS | Industrial Intelligence - Design System & CSS.
Enterprise-grade dark industrial IoT aesthetic inspired by SCADA/DCS monitoring platforms.
Features midnight navy surfaces, subtle cyan operating accents, amber threshold boundaries,
and vivid crimson anomaly alerts.
"""

AURALIS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --au-bg-canvas: #080C14;
    --au-bg-sidebar: #06090F;
    --au-bg-surface: #0E131F;
    --au-bg-surface-elevated: #131929;
    --au-bg-surface-hover: #182033;
    
    --au-border-subtle: rgba(255, 255, 255, 0.07);
    --au-border-medium: rgba(255, 255, 255, 0.12);
    --au-border-accent: rgba(6, 182, 212, 0.35);
    
    --au-cyan-primary: #06B6D4;
    --au-cyan-light: #22D3EE;
    --au-cyan-dim: rgba(6, 182, 212, 0.12);
    
    --au-emerald: #10B981;
    --au-emerald-dim: rgba(16, 185, 129, 0.12);
    
    --au-amber: #F59E0B;
    --au-amber-dim: rgba(245, 158, 11, 0.12);
    
    --au-red: #EF4444;
    --au-red-dim: rgba(239, 68, 68, 0.14);
    
    --au-text-primary: #F8FAFC;
    --au-text-secondary: #94A3B8;
    --au-text-muted: #64748B;
}

/* Global Container */
.stApp {
    background-color: var(--au-bg-canvas);
    color: var(--au-text-primary);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: var(--au-text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: var(--au-bg-sidebar) !important;
    border-right: 1px solid var(--au-border-subtle) !important;
}

/* Context / Telemetry Ribbon */
.context-ribbon {
    background-color: #0B101A;
    border: 1px solid var(--au-border-subtle);
    border-radius: 8px;
    padding: 10px 18px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}

.context-chips-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.context-chip {
    background-color: #121826;
    border: 1px solid var(--au-border-subtle);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--au-text-secondary);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.context-chip strong {
    color: var(--au-text-primary);
}

.context-chip.active-cyan {
    border-color: rgba(6, 182, 212, 0.4);
    color: var(--au-cyan-light);
    background-color: var(--au-cyan-dim);
}

.context-chip.active-green {
    border-color: rgba(16, 185, 129, 0.4);
    color: #34D399;
    background-color: var(--au-emerald-dim);
}

/* Metric / KPI Card */
.kpi-card {
    background-color: var(--au-bg-surface);
    border: 1px solid var(--au-border-subtle);
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}

.kpi-card:hover {
    border-color: var(--au-border-medium);
    background-color: var(--au-bg-surface-elevated);
}

.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--au-text-muted);
}

.kpi-icon-container {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--au-border-subtle);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--au-text-secondary);
}

.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--au-text-primary);
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.78rem;
    color: var(--au-text-secondary);
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Alert & Result Banners */
.industrial-verdict {
    border-radius: 8px;
    padding: 18px 22px;
    margin: 16px 0 22px 0;
    border: 1px solid var(--au-border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
}

.industrial-verdict.normal {
    background-color: rgba(6, 182, 212, 0.06);
    border-color: rgba(6, 182, 212, 0.3);
}

.industrial-verdict.anomaly {
    background-color: rgba(239, 68, 68, 0.07);
    border-color: rgba(239, 68, 68, 0.4);
}

.verdict-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 10px;
}

.verdict-title.normal {
    color: var(--au-cyan-light);
}

.verdict-title.anomaly {
    color: #F87171;
}

.verdict-desc {
    font-size: 0.88rem;
    color: #CBD5E1;
    margin-top: 6px;
    line-height: 1.5;
}

/* Clean Telemetry Badges */
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-cyan {
    background: var(--au-cyan-dim);
    border: 1px solid rgba(6, 182, 212, 0.35);
    color: var(--au-cyan-light);
}

.badge-green {
    background: var(--au-emerald-dim);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34D399;
}

.badge-amber {
    background: var(--au-amber-dim);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #FBBF24;
}

.badge-red {
    background: var(--au-red-dim);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #F87171;
}

/* Callout Notice */
.audit-notice {
    background: rgba(19, 25, 41, 0.6);
    border-left: 3px solid var(--au-cyan-primary);
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 18px 0;
    font-size: 0.85rem;
    color: #CBD5E1;
    line-height: 1.5;
}

.audit-notice.amber {
    border-left-color: var(--au-amber);
    background: rgba(245, 158, 11, 0.06);
}

.audit-notice.red {
    border-left-color: var(--au-red);
    background: rgba(239, 68, 68, 0.06);
}

/* Streamlit Widget Overrides */
div[data-testid="stExpander"] {
    background-color: var(--au-bg-surface) !important;
    border: 1px solid var(--au-border-subtle) !important;
    border-radius: 6px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #0A0E17;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid var(--au-border-subtle);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 4px;
    padding: 6px 14px;
    color: var(--au-text-secondary);
    font-size: 0.84rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background-color: #151C2C !important;
    color: var(--au-cyan-light) !important;
}

.stButton > button {
    background-color: #121826 !important;
    color: var(--au-text-primary) !important;
    border: 1px solid var(--au-border-medium) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    border-color: var(--au-cyan-primary) !important;
    color: #FFFFFF !important;
    background-color: #182236 !important;
}

.stButton > button[kind="primary"] {
    background-color: #0284C7 !important;
    border-color: #38BDF8 !important;
    color: #FFFFFF !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #0369A1 !important;
    border-color: #7DD3FC !important;
}

/* Dataframe clean styling */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--au-border-subtle);
    border-radius: 6px;
}
</style>
"""
