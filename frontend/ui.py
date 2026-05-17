import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
import requests
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalAI Lab Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #0a0d14; color: #e2e8f0; }

[data-testid="stSidebar"] { background: #0f1320 !important; border-right: 1px solid #1e2a3a; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

input[type="number"], input[type="text"], input[type="password"] {
    background: #131929 !important; border: 1px solid #1e2a3a !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    background: #1d4ed8 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #2563eb !important; }

[data-testid="metric-container"] {
    background: #131929; border: 1px solid #1e2a3a; border-radius: 12px; padding: 1rem 1.2rem !important;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; color: #f1f5f9 !important; }

.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }
hr { border-color: #1e2a3a !important; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; border-bottom: 1px solid #1e2a3a; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px 8px 0 0; color: #64748b; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #131929 !important; color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important; }

.badge-high   { background:#7f1d1d; color:#fca5a5; border:1px solid #f87171; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }
.badge-low    { background:#1e3a5f; color:#93c5fd; border:1px solid #60a5fa; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }
.badge-normal { background:#14532d; color:#86efac; border:1px solid #4ade80; padding:2px 10px; border-radius:20px; font-size:12px; font-weight:700; }

[data-testid="stNumberInput"] input { font-family: 'IBM Plex Mono', monospace !important; font-size:15px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
API_URL = "http://localhost:8000/api/predictions/analyze"

REFERENCE_RANGES = {
    "glucose":       {"min": 70,   "max": 99,   "unit": "mg/dL"},
    "cholesterol":   {"min": 0,    "max": 200,  "unit": "mg/dL"},
    "hdl":           {"min": 40,   "max": 999,  "unit": "mg/dL"},
    "ldl":           {"min": 0,    "max": 130,  "unit": "mg/dL"},
    "triglycerides": {"min": 0,    "max": 150,  "unit": "mg/dL"},
    "hemoglobin":    {"min": 12.0, "max": 17.5, "unit": "g/dL"},
    "creatinine":    {"min": 0.6,  "max": 1.2,  "unit": "mg/dL"},
    "bmi":           {"min": 18.5, "max": 24.9, "unit": ""},
    "systolic_bp":   {"min": 90,   "max": 120,  "unit": "mmHg"},
    "diastolic_bp":  {"min": 60,   "max": 80,   "unit": "mmHg"},
}

PARAM_LABELS = {
    "glucose": "Glucose", "cholesterol": "Cholesterol", "hdl": "HDL",
    "ldl": "LDL", "triglycerides": "Triglycerides", "hemoglobin": "Hemoglobin",
    "creatinine": "Creatinine", "bmi": "BMI",
    "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
}

RISK_COLORS = {
    "Low": "#4ade80", "Moderate": "#fbbf24", "High": "#f87171", "Critical": "#ef4444"
}

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "user_logged_in": False,
    "user_name": "",
    "user_dob": None,
    "analysis_result": None,
    "history": [],
    "last_values": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def flag_value(key: str, value: float) -> str:
    r = REFERENCE_RANGES.get(key)
    if r is None:
        return "normal"
    if value < r["min"]:
        return "low"
    if value > r["max"]:
        return "high"
    return "normal"

def risk_gauge(score: int, level: str):
    color = RISK_COLORS.get(level, "#94a3b8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#475569"}},
            "bar": {"color": color},
            "bgcolor": "#131929",
            "bordercolor": "#1e2a3a",
            "steps": [
                {"range": [0, 30],   "color": "#14532d"},
                {"range": [30, 60],  "color": "#78350f"},
                {"range": [60, 80],  "color": "#7c2d12"},
                {"range": [80, 100], "color": "#450a0a"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        },
        number={"font": {"color": color, "size": 36, "family": "IBM Plex Mono"}},
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
    )
    return fig

def trend_chart(param: str, current: float, d30: float, d60: float, d90: float):
    ref = REFERENCE_RANGES.get(param, {})
    df = pd.DataFrame({
        "Timepoint": ["Now", "+30d", "+60d", "+90d"],
        "Value": [current, d30, d60, d90],
    })
    fig = go.Figure()
    if ref:
        fig.add_hrect(
            y0=ref["min"], y1=ref["max"],
            fillcolor="rgba(34,197,94,0.08)", line_width=0,
            annotation_text="Normal range",
            annotation_font_size=10, annotation_font_color="#4ade80",
        )
    color      = "#ef4444" if d90 > current else "#3b82f6"
    fill_color = "rgba(239,68,68,0.07)" if d90 > current else "rgba(59,130,246,0.07)"
    fig.add_trace(go.Scatter(
        x=df["Timepoint"], y=df["Value"],
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color, line=dict(width=2, color="#0a0d14")),
        fill="tozeroy", fillcolor=fill_color,
    ))
    fig.update_layout(
        height=180, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#475569"),
        yaxis=dict(showgrid=True, gridcolor="#1e2a3a", color="#475569"),
        showlegend=False,
    )
    return fig

def biomarker_summary_chart(values: dict):
    params, pcts, colors, labels = [], [], [], []
    for k, v in values.items():
        r = REFERENCE_RANGES.get(k)
        if not r:
            continue
        rng  = r["max"] - r["min"] if r["max"] > r["min"] else 1
        mid  = (r["min"] + r["max"]) / 2
        deviation = (v - mid) / (rng / 2) * 100
        flag = flag_value(k, v)
        col  = "#4ade80" if flag == "normal" else ("#f87171" if flag == "high" else "#60a5fa")
        params.append(PARAM_LABELS.get(k, k))
        pcts.append(round(deviation, 1))
        colors.append(col)
        labels.append(f"{v} {r['unit']}")

    fig = go.Figure(go.Bar(
        x=pcts, y=params, orientation="h",
        marker_color=colors, text=labels,
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8", family="IBM Plex Mono"),
    ))
    fig.add_vline(x=0, line_color="#475569", line_width=1, line_dash="dot")
    fig.add_vrect(x0=-100, x1=100, fillcolor="rgba(34,197,94,0.04)", line_width=0)
    fig.update_layout(
        height=340, margin=dict(t=10, b=10, l=0, r=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1e2a3a", color="#475569",
                   ticksuffix="%", title="Deviation from midpoint of normal range"),
        yaxis=dict(showgrid=False, color="#94a3b8"),
        bargap=0.3,
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1rem'>
      <div style='font-size:1.5rem;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px'>🧬 ClinicalAI</div>
      <div style='font-size:0.72rem;font-weight:600;color:#3b82f6;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px'>Lab Analyzer · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.user_logged_in:
        st.markdown(f"""
        <div style='background:#131929;border:1px solid #1e2a3a;border-radius:10px;padding:10px 14px;margin-bottom:1rem'>
          <div style='font-size:11px;color:#64748b;font-weight:600;letter-spacing:0.08em'>SIGNED IN AS</div>
          <div style='font-size:15px;color:#f1f5f9;font-weight:700;margin-top:2px'>{st.session_state.user_name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    selection = st.radio(
        "Navigation",
        ["🔐  Login", "🔬  Analyze", "📋  History", "📊  Reference"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#475569;line-height:1.7'>
    <b style='color:#64748b'>HOW IT WORKS</b><br>
    1 · Login with your credentials<br>
    2 · Enter biomarker values<br>
    3 · AI analyses risk + trends<br>
    4 · Review recommendations
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:10px;color:#334155;border-top:1px solid #1e2a3a;padding-top:0.8rem'>
    ⚕️ For clinical decision support only.<br>Not a substitute for professional diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: LOGIN
# ─────────────────────────────────────────────
if "Login" in selection:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width:460px;margin:0 auto'>
      <h1 style='color:#f1f5f9;margin-bottom:4px'>Welcome back</h1>
      <p style='color:#64748b;font-size:15px;margin-bottom:2rem'>Sign in to access your lab analysis dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        with st.form("login_form"):
            st.markdown("**Full Name**")
            name = st.text_input("Full Name", placeholder="e.g. Dr. Arjun Mehta", label_visibility="collapsed")
            st.markdown("**Password**")
            password = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")
            st.markdown("**Confirm Password**")
            confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", label_visibility="collapsed")
            st.markdown("**Date of Birth**")
            dob = st.date_input("Date of Birth", min_value=date(1920, 1, 1), max_value=date.today(), label_visibility="collapsed")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
            elif len(password) < 4:
                st.error("Password must be at least 4 characters.")
            elif password != confirm:
                st.error("Passwords don't match.")
            else:
                st.session_state.user_logged_in = True
                st.session_state.user_name = name.strip()
                st.session_state.user_dob = dob
                st.success(f"✅ Welcome, {name.strip()}! Navigate to Analyze to begin.")
                time.sleep(0.8)
                st.rerun()

# ─────────────────────────────────────────────
#  PAGE: ANALYZE
# ─────────────────────────────────────────────
elif "Analyze" in selection:
    if not st.session_state.user_logged_in:
        st.warning("🔐 Please sign in first using the Login page.")
        st.stop()

    st.markdown("""
    <h1 style='color:#f1f5f9;margin-bottom:2px'>Lab Values Analysis</h1>
    <p style='color:#64748b;font-size:14px;margin-bottom:1.5rem'>Enter biomarker readings. Flagged values are highlighted against clinical reference ranges.</p>
    """, unsafe_allow_html=True)

    st.markdown("#### Patient & Metabolic")
    c1, c2, c3 = st.columns(3)
    with c1:
        age           = st.number_input("Age (years)",           0,   150,  55,  1)
        glucose       = st.number_input("Glucose (mg/dL)",       0,   500, 150,  1)
        cholesterol   = st.number_input("Cholesterol (mg/dL)",   0,   500, 220,  1)
        hdl           = st.number_input("HDL (mg/dL)",           0,   200,  35,  1)
    with c2:
        ldl           = st.number_input("LDL (mg/dL)",           0,   500, 160,  1)
        triglycerides = st.number_input("Triglycerides (mg/dL)", 0,   500, 180,  1)
        hemoglobin    = st.number_input("Hemoglobin (g/dL)",     5.0, 20.0, 13.2, 0.1)
        creatinine    = st.number_input("Creatinine (mg/dL)",    0.0, 10.0,  1.1, 0.1)
    with c3:
        bmi           = st.number_input("BMI",                   10.0, 60.0, 28.0, 0.1)
        systolic_bp   = st.number_input("Systolic BP (mmHg)",    60,  250, 135,   1)
        diastolic_bp  = st.number_input("Diastolic BP (mmHg)",   40,  150,  85,   1)

    # Live flag summary
    values_dict = {
        "glucose": glucose, "cholesterol": cholesterol, "hdl": hdl,
        "ldl": ldl, "triglycerides": triglycerides, "hemoglobin": hemoglobin,
        "creatinine": creatinine, "bmi": bmi,
        "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
    }
    abnormal = [(k, v, flag_value(k, v)) for k, v in values_dict.items() if flag_value(k, v) != "normal"]

    if abnormal:
        icons = {"high": "↑ HIGH", "low": "↓ LOW"}
        flags_html = " &nbsp;".join(
            f'<span class="badge-{f}">{PARAM_LABELS[k]}: {v} {REFERENCE_RANGES[k]["unit"]} {icons.get(f,"")}</span>'
            for k, v, f in abnormal
        )
        st.markdown(f"<div style='margin:0.5rem 0 1rem;line-height:2.4'>{flags_html}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#4ade80;font-size:13px;margin-bottom:1rem'>✓ All values within reference ranges</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        analyze_clicked = st.button("🔍 Run AI Analysis", type="primary", use_container_width=True)

    if analyze_clicked:
        lab_values = {
            "age": float(age), "glucose": float(glucose),
            "cholesterol": float(cholesterol), "hdl": float(hdl),
            "ldl": float(ldl), "triglycerides": float(triglycerides),
            "hemoglobin": float(hemoglobin), "creatinine": float(creatinine),
            "bmi": float(bmi), "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
        }
        st.session_state.last_values = lab_values

        with st.spinner("Running multi-agent AI analysis…"):
            try:
                resp = requests.post(API_URL, json=lab_values, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    result["_timestamp"] = datetime.now().strftime("%d %b %Y, %H:%M")
                    result["_values"] = lab_values
                    st.session_state.analysis_result = result
                    st.session_state.history.insert(0, {
                        "timestamp": result["_timestamp"],
                        "values": lab_values,
                        "result": result,
                    })
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[:10]
                else:
                    st.error(f"API returned status {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach the backend at `localhost:8000`. Make sure it's running:\n```\nuvicorn backend.app.main:app --reload\n```")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out after 30 seconds.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # ── Results ──
    if st.session_state.analysis_result:
        result  = st.session_state.analysis_result
        agent2  = result.get("agent2_disease_detection", {})
        agent3  = result.get("agent3_trend_forecasting", {})
        agent4  = result.get("agent4_risk_assessment", {})
        vals    = result.get("_values", {})
        ts      = result.get("_timestamp", "")

        risk_level = agent4.get("risk_level", "N/A")
        risk_score = int(agent4.get("risk_score", 0))
        risk_color = RISK_COLORS.get(risk_level, "#94a3b8")

        st.markdown("---")
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:1.2rem'>
          <h1 style='margin:0;color:#f1f5f9;font-size:1.4rem'>Analysis Results</h1>
          <span style='font-size:12px;color:#475569;font-family:IBM Plex Mono'>{ts}</span>
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(["📊 Overview", "📈 Trends Forecast", "🩺 Recommendations", "🧪 Biomarker Detail"])

        # Tab 1: Overview
        with tabs[0]:
            st.markdown("<br>", unsafe_allow_html=True)
            col_g, col_m = st.columns([1, 2])
            with col_g:
                st.markdown(f"<div style='text-align:center;margin-bottom:0.3rem'><span style='font-size:11px;font-weight:700;color:{risk_color};letter-spacing:0.12em'>{risk_level.upper()} RISK</span></div>", unsafe_allow_html=True)
                st.plotly_chart(risk_gauge(risk_score, risk_level), use_container_width=True, config={"displayModeBar": False})
            with col_m:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Disease Flag",  agent2.get("disease_type", "N/A"))
                m2.metric("Risk Score",    f"{risk_score}/100")
                m3.metric("Confidence",    f"{agent4.get('confidence', 0)*100:.0f}%")
                m4.metric("Process Time",  f"{result.get('processing_time_ms', 0):.0f} ms")
                st.markdown("<br>", unsafe_allow_html=True)
                if vals:
                    st.markdown("**Deviation from normal range midpoint**")
                    st.plotly_chart(biomarker_summary_chart(vals), use_container_width=True, config={"displayModeBar": False})

        # Tab 2: Trends
        with tabs[1]:
            if not agent3:
                st.info("No trend data returned by the API.")
            else:
                valid = {k: v for k, v in agent3.items() if "error" not in v}
                if not valid:
                    st.warning("All trend forecasts returned errors.")
                else:
                    st.markdown("<br>", unsafe_allow_html=True)
                    params_list = list(valid.items())
                    for i in range(0, len(params_list), 2):
                        row = params_list[i: i + 2]
                        cols = st.columns(len(row))
                        for col, (param, forecast) in zip(cols, row):
                            with col:
                                current = forecast.get("current", 0)
                                d30     = forecast.get("forecast_30d", 0)
                                d60     = forecast.get("forecast_60d", 0)
                                d90     = forecast.get("forecast_90d", 0)
                                trend   = forecast.get("trend", "N/A")
                                unit    = REFERENCE_RANGES.get(param, {}).get("unit", "")
                                trend_icon = "↑" if "incr" in trend.lower() or "up" in trend.lower() \
                                             else "↓" if "decr" in trend.lower() or "down" in trend.lower() else "→"

                                st.markdown(f"""
                                <div style='background:#131929;border:1px solid #1e2a3a;border-radius:10px;padding:14px 16px;margin-bottom:6px'>
                                  <div style='font-size:13px;font-weight:700;color:#94a3b8;letter-spacing:0.05em;text-transform:uppercase'>{PARAM_LABELS.get(param, param)}</div>
                                  <div style='font-family:IBM Plex Mono;font-size:22px;color:#f1f5f9;margin:4px 0 2px'>{current} <span style='font-size:12px;color:#475569'>{unit}</span></div>
                                  <div style='font-size:12px;color:#64748b'>Trend: <b style='color:#cbd5e1'>{trend_icon} {trend}</b></div>
                                </div>
                                """, unsafe_allow_html=True)

                                st.plotly_chart(
                                    trend_chart(param, current, d30, d60, d90),
                                    use_container_width=True,
                                    config={"displayModeBar": False},
                                )

                                st.markdown(f"""
                                <div style='display:flex;gap:8px;font-family:IBM Plex Mono;font-size:12px;color:#64748b;margin-bottom:1rem'>
                                  <span>+30d: <b style='color:#94a3b8'>{d30} {unit}</b></span>
                                  <span>+60d: <b style='color:#94a3b8'>{d60} {unit}</b></span>
                                  <span>+90d: <b style='color:#94a3b8'>{d90} {unit}</b></span>
                                </div>
                                """, unsafe_allow_html=True)

        # Tab 3: Recommendations
        with tabs[2]:
            recs = agent4.get("recommendations", [])
            if not recs:
                st.info("No recommendations returned.")
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                for i, rec in enumerate(recs, 1):
                    priority_color = "#f87171" if i == 1 else "#fbbf24" if i == 2 else "#60a5fa"
                    st.markdown(f"""
                    <div style='background:#131929;border:1px solid #1e2a3a;border-left:3px solid {priority_color};
                                border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px;display:flex;gap:12px;align-items:flex-start'>
                      <span style='font-size:11px;font-weight:700;color:{priority_color};min-width:20px;padding-top:1px'>#{i}</span>
                      <span style='font-size:14px;color:#cbd5e1;line-height:1.5'>{rec}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 16px;font-size:12px;color:#64748b'>
            ⚕️ <b style='color:#94a3b8'>Clinical Disclaimer</b> — These insights are generated by AI for decision-support purposes only.
            They do not constitute a medical diagnosis. Always consult a qualified healthcare professional before acting on these results.
            </div>
            """, unsafe_allow_html=True)

        # Tab 4: Biomarker Detail
        with tabs[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            if not vals:
                st.info("No values to display.")
            else:
                rows = []
                for k, v in vals.items():
                    if k == "age":
                        continue
                    r    = REFERENCE_RANGES.get(k, {})
                    flag = flag_value(k, v)
                    rows.append({
                        "Biomarker": PARAM_LABELS.get(k, k),
                        "Value":     v,
                        "Unit":      r.get("unit", ""),
                        "Min":       r.get("min", "—"),
                        "Max":       r.get("max", "—"),
                        "Status":    flag.upper(),
                    })

                def color_status(val):
                    c = {"HIGH": "color:#f87171", "LOW": "color:#60a5fa", "NORMAL": "color:#4ade80"}
                    return c.get(val, "")

                st.dataframe(
                    pd.DataFrame(rows).style.map(color_status, subset=["Status"]),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_new, _ = st.columns([2, 5])
        with col_new:
            if st.button("↩ New Analysis", use_container_width=True):
                st.session_state.analysis_result = None
                st.rerun()

# ─────────────────────────────────────────────
#  PAGE: HISTORY
# ─────────────────────────────────────────────
elif "History" in selection:
    if not st.session_state.user_logged_in:
        st.warning("🔐 Please sign in first.")
        st.stop()

    st.markdown("""
    <h1 style='color:#f1f5f9;margin-bottom:4px'>Analysis History</h1>
    <p style='color:#64748b;font-size:14px;margin-bottom:1.5rem'>Your 10 most recent analyses this session.</p>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Run your first analysis on the Analyze page.")
    else:
        for entry in st.session_state.history:
            r4    = entry["result"].get("agent4_risk_assessment", {})
            r2    = entry["result"].get("agent2_disease_detection", {})
            level = r4.get("risk_level", "N/A")
            score = r4.get("risk_score", 0)

            with st.expander(f"🕐 {entry['timestamp']}  ·  Risk: {level} ({score}/100)  ·  {r2.get('disease_type', 'N/A')}"):
                vals = entry.get("values", {})
                if vals:
                    cols = st.columns(5)
                    for i, (k, v) in enumerate(vals.items()):
                        with cols[i % 5]:
                            ref  = REFERENCE_RANGES.get(k, {})
                            unit = ref.get("unit", "")
                            st.metric(PARAM_LABELS.get(k, k), f"{v} {unit}")
                recs = r4.get("recommendations", [])
                if recs:
                    st.markdown("**Recommendations:**")
                    for rec in recs:
                        st.markdown(f"• {rec}")

# ─────────────────────────────────────────────
#  PAGE: REFERENCE RANGES
# ─────────────────────────────────────────────
elif "Reference" in selection:
    st.markdown("""
    <h1 style='color:#f1f5f9;margin-bottom:4px'>Clinical Reference Ranges</h1>
    <p style='color:#64748b;font-size:14px;margin-bottom:1.5rem'>Adult reference ranges used by the analyzer for flagging.</p>
    """, unsafe_allow_html=True)

    interpretations = {
        "glucose":       "Fasting blood glucose. >126 mg/dL on two occasions = diabetes.",
        "cholesterol":   "Total cholesterol. >240 mg/dL = high risk for CVD.",
        "hdl":           "Good cholesterol. <40 mg/dL increases CVD risk.",
        "ldl":           "Bad cholesterol. >160 mg/dL warrants dietary intervention.",
        "triglycerides": ">500 mg/dL = severe hypertriglyceridaemia, pancreatitis risk.",
        "hemoglobin":    "Low = anaemia; high = polycythaemia.",
        "creatinine":    "Marker of kidney function. Elevated → CKD screening needed.",
        "bmi":           "18.5–24.9 healthy; ≥30 = obesity.",
        "systolic_bp":   "≥130 mmHg = Stage 1 hypertension (AHA 2017).",
        "diastolic_bp":  "≥80 mmHg = Stage 1 hypertension.",
    }

    ref_data = []
    for k, r in REFERENCE_RANGES.items():
        ref_data.append({
            "Biomarker":     PARAM_LABELS.get(k, k),
            "Min":           r["min"] if r["min"] > 0 else "—",
            "Max":           r["max"] if r["max"] < 900 else "No upper limit",
            "Unit":          r["unit"],
            "Clinical Note": interpretations.get(k, ""),
        })

    st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

    st.markdown("""
    <br>
    <div style='background:#131929;border:1px solid #1e2a3a;border-radius:10px;padding:14px 18px;font-size:13px;color:#64748b'>
    Sources: AHA/ACC guidelines, WHO reference values, ICMR adult reference ranges (India).
    Values may vary by laboratory, sex, and age. These are population-level estimates for screening purposes.
    </div>
    """, unsafe_allow_html=True)