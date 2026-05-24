import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor · Pooja Dhamale",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS — premium dark theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --gold: #FBBF24;
    --gold-dim: rgba(251,191,36,.12);
    --bg: #0A0A0A;
    --card: #111111;
    --card-hover: #161616;
    --border: rgba(255,255,255,.07);
    --text: #F1F5F9;
    --muted: #94A3B8;
    --green: #34D399;
    --red: #F87171;
}

html, body, [class*="st-"] { font-family:'Inter',sans-serif; }
.stApp { background: var(--bg); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0F0F0F 0%,#0A0A0A 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio > label { display:none; }
section[data-testid="stSidebar"] .stRadio > div {
    display:flex; flex-direction:column; gap:2px;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding:10px 16px !important; border-radius:10px !important;
    font-weight:500 !important; font-size:.92rem !important;
    transition:all .2s ease !important; cursor:pointer !important;
    color:var(--muted) !important; border:1px solid transparent !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background:var(--gold-dim) !important; color:var(--gold) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
    color:var(--text) !important;
}

/* ── Cards ── */
.glass-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:16px; padding:28px; margin-bottom:20px;
    transition:transform .2s ease, box-shadow .2s ease;
}
.glass-card:hover { transform:translateY(-2px); box-shadow:0 8px 30px rgba(0,0,0,.4); }

.kpi-card {
    background:var(--card); border:1px solid var(--border);
    border-radius:14px; padding:22px 20px; text-align:center;
}
.kpi-icon { font-size:1.6rem; margin-bottom:6px; }
.kpi-label { font-size:.78rem; color:var(--muted); font-weight:500; letter-spacing:.4px; text-transform:uppercase; }
.kpi-value { font-size:1.8rem; font-weight:700; color:var(--gold); margin:4px 0; }
.kpi-sub { font-size:.75rem; color:var(--muted); }

/* ── Section headers ── */
.section-header {
    font-size:1.5rem; font-weight:700; color:var(--text);
    margin:2rem 0 1rem 0; display:flex; align-items:center; gap:10px;
}
.section-header span { font-size:1.4rem; }

.tag { display:inline-block; padding:4px 12px; border-radius:20px;
       font-size:.72rem; font-weight:600; letter-spacing:.3px; }
.tag-gold { background:var(--gold-dim); color:var(--gold); }
.tag-green { background:rgba(52,211,153,.12); color:#34D399; }
.tag-red { background:rgba(248,113,113,.12); color:#F87171; }

/* ── Metric badge ── */
.metric-row { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; }
.metric-badge {
    background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:10px 18px;
    display:flex; flex-direction:column; align-items:center;
}
.metric-badge .label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.3px; }
.metric-badge .value { font-size:1.1rem; font-weight:700; color:var(--gold); }

/* ── Pipeline steps ── */
.pipeline { display:flex; gap:0; align-items:center; flex-wrap:wrap; justify-content:center; margin:20px 0; }
.pipe-step {
    background:var(--card); border:1px solid var(--border);
    border-radius:12px; padding:14px 20px; text-align:center; min-width:140px;
}
.pipe-step .num { font-size:.65rem; color:var(--gold); font-weight:700; letter-spacing:1px; }
.pipe-step .name { font-size:.85rem; font-weight:600; color:var(--text); margin-top:2px; }
.pipe-arrow { font-size:1.2rem; color:var(--muted); margin:0 4px; }

/* ── Inputs ── */
div[data-baseweb="input"], div[data-baseweb="select"], .stSelectbox > div,
.stNumberInput > div > div { background:#1A1A1A !important; border:1px solid var(--border) !important; border-radius:10px !important; }
input, select, textarea { color:var(--text) !important; }
div[role="listbox"] { background:#1A1A1A !important; }
div[role="option"]:hover { background:var(--gold) !important; color:#000 !important; }
label { color:var(--muted) !important; font-weight:500 !important; }

/* ── Button ── */
.stButton>button {
    background:linear-gradient(135deg,#FBBF24,#F59E0B) !important;
    color:#000 !important; font-weight:700 !important; border:none !important;
    border-radius:12px !important; padding:12px 28px !important;
    font-size:.95rem !important; transition:all .3s ease !important;
    box-shadow:0 4px 15px rgba(251,191,36,.25) !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 25px rgba(251,191,36,.4) !important;
}

/* ── Plotly charts ── */
.js-plotly-plot .plotly .modebar { display:none !important; }

h1,h2,h3,h4 { color:var(--text) !important; }
a { color:var(--gold) !important; }

/* ── Footer ── */
.footer { text-align:center; padding:30px 0 10px 0; color:var(--muted); font-size:.8rem; border-top:1px solid var(--border); margin-top:40px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    return pd.read_csv("indian_bank_customer_churn.csv")

@st.cache_data
def load_featured():
    return pd.read_csv("indian_bank_customer_churn_featured.csv")

@st.cache_resource
def load_model():
    model = pickle.load(open("xgb_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoder = pickle.load(open("encoder.pkl", "rb"))
    features = pickle.load(open("model_features.pkl", "rb"))
    return model, scaler, encoder, features

df = load_raw()
df_feat = load_featured()
model, scaler, encoder, feature_names = load_model()

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
        <div style="font-size:2.2rem;">🏦</div>
        <div style="font-size:1.1rem;font-weight:700;color:#FBBF24;margin-top:4px;">Bank Churn Predictor</div>
        <div style="font-size:.72rem;color:#94A3B8;margin-top:2px;">End-to-End ML Project</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    section = st.radio(
        "Navigate",
        ["🏠 Project Overview", "📊 EDA & Insights", "⚙️ ML Pipeline", "🔮 Live Prediction", "📌 Business Strategy"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <div style="font-size:.72rem;color:#64748B;margin-bottom:6px;">BUILT BY</div>
        <div style="font-size:.9rem;font-weight:600;color:#F1F5F9;">Pooja Dhamale</div>
        <div style="font-size:.72rem;color:#94A3B8;">ML & Data Science</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════
if section == "🏠 Project Overview":
    st.markdown("""
    <div style="text-align:center;padding:30px 0 10px 0;">
        <span class="tag tag-gold">END-TO-END MACHINE LEARNING</span>
        <h1 style="font-size:2.6rem;font-weight:800;margin:16px 0 6px 0;">
            🛡️ Indian Bank Customer<br>Churn Prediction System
        </h1>
        <p style="color:#94A3B8;font-size:1.05rem;max-width:650px;margin:0 auto;">
            Predicting customer attrition using <b style="color:#FBBF24;">XGBoost</b> on 100K banking records
            with engineered behavioral features — optimized for <b style="color:#34D399;">90% Recall</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    total = len(df)
    churned = df["Churn"].sum()
    retained = total - churned
    churn_rate = churned / total * 100

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("👥", "Total Customers", f"{total:,}", "100K records"),
        ("❌", "Churned", f"{churned:,}", f"{churn_rate:.1f}% attrition"),
        ("✅", "Retained", f"{retained:,}", f"{100-churn_rate:.1f}% retained"),
        ("🎯", "Model Recall", "90.3%", "Churn class (threshold 0.3)"),
    ]
    for col, (icon, label, value, sub) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="section-header"><span>⚙️</span> ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        <div class="pipe-step"><div class="num">STEP 01</div><div class="name">Data Collection</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">STEP 02</div><div class="name">EDA</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">STEP 03</div><div class="name">Feature Eng.</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">STEP 04</div><div class="name">Model Training</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">STEP 05</div><div class="name">Evaluation</div></div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step"><div class="num">STEP 06</div><div class="name">Deployment</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement & Tech
    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#FBBF24 !important;margin-top:0;">📌 Problem Statement</h3>
            <p style="color:#CBD5E1;line-height:1.7;">
            Customer churn is a critical challenge for Indian banks — acquiring a new customer costs
            <b style="color:#F87171;">5-7× more</b> than retaining an existing one.
            This project builds a predictive system that identifies at-risk customers <i>before</i>
            they leave, enabling proactive retention strategies.
            </p>
            <div style="margin-top:12px;">
                <span class="tag tag-gold">Classification</span>&nbsp;
                <span class="tag tag-green">Supervised ML</span>&nbsp;
                <span class="tag tag-red">Imbalanced Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#FBBF24 !important;margin-top:0;">🛠️ Tech Stack</h3>
            <table style="width:100%;color:#CBD5E1;font-size:.88rem;">
                <tr><td style="padding:6px 0;color:#94A3B8;width:40%;">Language</td><td style="font-weight:600;">Python 3.10</td></tr>
                <tr><td style="padding:6px 0;color:#94A3B8;">ML Framework</td><td style="font-weight:600;">XGBoost, Scikit-Learn</td></tr>
                <tr><td style="padding:6px 0;color:#94A3B8;">Data</td><td style="font-weight:600;">Pandas, NumPy</td></tr>
                <tr><td style="padding:6px 0;color:#94A3B8;">Visualization</td><td style="font-weight:600;">Plotly, Seaborn</td></tr>
                <tr><td style="padding:6px 0;color:#94A3B8;">Deployment</td><td style="font-weight:600;">Streamlit</td></tr>
                <tr><td style="padding:6px 0;color:#94A3B8;">Tuning</td><td style="font-weight:600;">RandomizedSearchCV</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Dataset preview
    st.markdown('<div class="section-header"><span>📋</span> Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# SECTION 2 — EDA & INSIGHTS
# ══════════════════════════════════════════════════════════════
elif section == "📊 EDA & Insights":
    st.markdown('<div class="section-header"><span>📊</span> Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#CBD5E1;margin:0;">
        Visual exploration of <b>100,000 customer records</b> to uncover churn patterns across
        demographics, geography, and account behavior — <i>before</i> any feature engineering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_eda = df.copy()
    df_eda["Status"] = df_eda["Churn"].map({0: "Retained", 1: "Churned"})
    colors = {"Retained": "#64748B", "Churned": "#FBBF24"}
    layout_cfg = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#CBD5E1", family="Inter"), margin=dict(t=40, b=30, l=40, r=20))

    # Row 1 — Churn Distribution + Gender
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        counts = df_eda["Status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count", hole=0.55, color="Status",
                     color_discrete_map=colors, title="Overall Churn Distribution")
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#0A0A0A", width=2)))
        fig.update_layout(**layout_cfg, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        fig = px.histogram(df_eda, x="Gender", color="Status", barmode="group",
                           color_discrete_map=colors, title="Churn by Gender")
        fig.update_layout(**layout_cfg, xaxis_title="Gender", yaxis_title="Customers",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — State + Active Member
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = px.histogram(df_eda, x="State", color="Status", barmode="group",
                           color_discrete_map=colors, title="Churn by State")
        fig.update_layout(**layout_cfg, xaxis_title="State", yaxis_title="Customers",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        df_eda["Active"] = df_eda["Is_Active_Member"].map({0: "Inactive", 1: "Active"})
        fig = px.histogram(df_eda, x="Active", color="Status", barmode="group",
                           color_discrete_map=colors, title="Churn vs Activity Status")
        fig.update_layout(**layout_cfg, xaxis_title="Member Status", yaxis_title="Customers",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — Credit Score + Age Distribution
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = px.box(df_eda, x="Status", y="Credit_Score", color="Status",
                     color_discrete_map=colors, title="Credit Score Distribution")
        fig.update_layout(**layout_cfg, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        fig = px.histogram(df_eda, x="Age", color="Status", barmode="overlay", nbins=30,
                           color_discrete_map=colors, title="Age Distribution by Churn", opacity=0.7)
        fig.update_layout(**layout_cfg, xaxis_title="Age", yaxis_title="Count",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    # Churn Rate Cards
    st.markdown('<div class="section-header"><span>📈</span> Churn Rate Breakdown</div>', unsafe_allow_html=True)
    cr_gender = df.groupby("Gender")["Churn"].mean() * 100
    cr_active = df.groupby("Is_Active_Member")["Churn"].mean() * 100

    cols = st.columns(4)
    rate_data = [
        ("♀️ Female", f"{cr_gender.get('Female', 0):.1f}%"),
        ("♂️ Male", f"{cr_gender.get('Male', 0):.1f}%"),
        ("🟢 Active", f"{cr_active.get(1, 0):.1f}%"),
        ("🔴 Inactive", f"{cr_active.get(0, 0):.1f}%"),
    ]
    for col, (lbl, val) in zip(cols, rate_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{lbl}</div>
                <div class="kpi-value" style="font-size:1.5rem;">{val}</div>
                <div class="kpi-sub">churn rate</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — ML PIPELINE
# ══════════════════════════════════════════════════════════════
elif section == "⚙️ ML Pipeline":
    st.markdown('<div class="section-header"><span>⚙️</span> Machine Learning Pipeline</div>', unsafe_allow_html=True)

    # Feature Engineering
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#FBBF24 !important;margin-top:0;">🧠 Feature Engineering</h3>
        <p style="color:#CBD5E1;margin-bottom:16px;">Four domain-driven features were created to capture hidden customer behavior:</p>
        <table style="width:100%;border-collapse:collapse;color:#CBD5E1;font-size:.88rem;">
            <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
                <td style="padding:10px 12px;font-weight:600;color:#FBBF24;width:35%;">Balance_to_Salary</td>
                <td style="padding:10px 12px;">Ratio of account balance to salary — financial dependency indicator</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
                <td style="padding:10px 12px;font-weight:600;color:#FBBF24;">Tenure_NumProducts</td>
                <td style="padding:10px 12px;">Tenure × Products — depth of customer relationship</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(255,255,255,.07);">
                <td style="padding:10px 12px;font-weight:600;color:#FBBF24;">Low_Credit_Score</td>
                <td style="padding:10px 12px;">Binary flag for Credit Score &lt; 600 — financial risk</td>
            </tr>
            <tr>
                <td style="padding:10px 12px;font-weight:600;color:#FBBF24;">HighBalance_LowActivity</td>
                <td style="padding:10px 12px;">High balance + inactive member — strong exit intent signal</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison
    st.markdown('<div class="section-header"><span>🤖</span> Model Comparison</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("""
        <div class="glass-card" style="border-left:3px solid #64748B;">
            <h4 style="margin-top:0;">🌲 Random Forest <span class="tag tag-gold" style="font-size:.65rem;">BASELINE</span></h4>
            <p style="color:#94A3B8;font-size:.85rem;">n_estimators=200, random_state=42</p>
            <div class="metric-row">
                <div class="metric-badge">
                    <span class="label">Accuracy</span>
                    <span class="value">64.72%</span>
                </div>
                <div class="metric-badge">
                    <span class="label">Recall</span>
                    <span class="value">36%</span>
                </div>
            </div>
            <div class="metric-row" style="margin-top:10px;">
                <div class="metric-badge">
                    <span class="label">Precision</span>
                    <span class="value">55%</span>
                </div>
                <div class="metric-badge">
                    <span class="label">F1-Score</span>
                    <span class="value">44%</span>
                </div>
            </div>
            <p style="color:#F87171;font-size:.82rem;margin-bottom:0;">
                ⚠️ Missed 4,828 actual churn customers (High False Negatives)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown("""
        <div class="glass-card" style="border-left:3px solid #FBBF24;">
            <h4 style="margin-top:0;">⚡ XGBoost <span class="tag tag-green" style="font-size:.65rem;">FINAL SELECTED MODEL</span></h4>
            <p style="color:#94A3B8;font-size:.85rem;">
                Hyperparameter Tuned using RandomizedSearchCV<br>
                Optimized for Recall (Business Objective)
            </p>
            <div class="metric-row">
                <div class="metric-badge">
                    <span class="label">Recall</span>
                    <span class="value" style="color:#34D399;">90%</span>
                </div>
                <div class="metric-badge">
                    <span class="label">Precision</span>
                    <span class="value">42%</span>
                </div>
            </div>
            <div class="metric-row" style="margin-top:10px;">
                <div class="metric-badge">
                    <span class="label">Accuracy</span>
                    <span class="value">49.47%</span>
                </div>
                <div class="metric-badge">
                    <span class="label">F1-Score</span>
                    <span class="value">58%</span>
                </div>
            </div>
            <p style="color:#34D399;font-size:.82rem;margin-bottom:0;">
                ✅ Successfully detected 6,845 out of 7,578 churn customers
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Confusion Matrix + Feature Importance
    fi_left, fi_right = st.columns(2)

    with fi_left:
        st.markdown(
            '<div class="section-header" style="font-size:1.2rem;"><span>🧮</span> Confusion Matrix (Final XGBoost)</div>',
            unsafe_allow_html=True
        )

        cm = np.array([
            [3049, 9373],
            [733,  6845]
        ])

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Retained", "Predicted Churned"],
            y=["Actually Retained", "Actually Churned"],
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale=[[0, "#1a1a2e"], [1, "#FBBF24"]],
            showscale=False
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", family="Inter"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig, use_container_width=True)

    with fi_right:
        st.markdown(
            '<div class="section-header" style="font-size:1.2rem;"><span>📊</span> Feature Importance (Top 10)</div>',
            unsafe_allow_html=True
        )

        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)

        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#334155", "#FBBF24"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", family="Inter"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=350,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key Business Insight
    st.markdown("""
    <div class="glass-card" style="border-left:3px solid #34D399;">
        <h4 style="color:#34D399 !important;margin-top:0;">💡 Business-Oriented Model Selection</h4>
        <p style="color:#CBD5E1;margin-bottom:0;line-height:1.7;">
        The final XGBoost model was intentionally optimized for
        <b>high Recall (90%)</b> because the primary business goal is to
        identify customers likely to churn.
        Although the overall accuracy decreased to <b>49.47%</b>, the model
        dramatically reduced False Negatives from <b>4,828</b> (Random Forest)
        to just <b>733</b>.
        In customer churn prediction, missing a real churn customer can lead
        to direct revenue loss, whereas incorrectly flagging a loyal customer
        only results in a minor retention cost.
        Therefore, prioritizing Recall over Accuracy is a deliberate
        business-aligned machine learning strategy.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 4 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
elif section == "🔮 Live Prediction":
    st.markdown('<div class="section-header"><span>🔮</span> Live Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#CBD5E1;margin:0;">Enter customer details to get a <b style="color:#FBBF24;">real-time churn probability</b>
        from the trained XGBoost model.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("🎂 Age", 18, 100, 35)
            tenure = st.number_input("📆 Tenure (Years)", 0, 14, 5)
            credit_score = st.number_input("💳 Credit Score", 300, 900, 650)
            gender = st.selectbox("🧑 Gender", ["Male", "Female"])
        with col2:
            balance = st.number_input("💰 Balance (₹)", 0.0, 1_000_000.0, 50000.0, step=1000.0)
            salary = st.number_input("💼 Salary (₹)", 100_000.0, 3_000_000.0, 600000.0, step=10000.0)
            num_products = st.selectbox("📦 Products", [1, 2, 3, 4])
            state = st.selectbox("📍 State", sorted(df["State"].unique()))
        with col3:
            has_cc = st.selectbox("💳 Credit Card?", ["Yes", "No"])
            active = st.selectbox("⚡ Active Member?", ["Yes", "No"])
            account_type = st.selectbox("🏦 Account Type", sorted(df["Account_Type"].unique()))

    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

    if predict_btn:
        has_cc_val = 1 if has_cc == "Yes" else 0
        active_val = 1 if active == "Yes" else 0
        balance_to_salary = balance / salary if salary > 0 else 0
        tenure_numproducts = tenure * num_products
        low_credit_score = 1 if credit_score < 600 else 0
        highbalance_lowactivity = 1 if (balance > 100000 and active_val == 0) else 0

        input_df = pd.DataFrame({
            "Age": [age], "Tenure_Years": [tenure], "Balance_INR": [balance],
            "Num_Products": [num_products], "Has_Credit_Card": [has_cc_val],
            "Is_Active_Member": [active_val], "Estimated_Salary_INR": [salary],
            "Credit_Score": [credit_score], "Balance_to_Salary": [balance_to_salary],
            "Tenure_NumProducts": [tenure_numproducts], "Low_Credit_Score": [low_credit_score],
            "HighBalance_LowActivity": [highbalance_lowactivity],
            "Gender": [gender], "State": [state], "Account_Type": [account_type]
        })

        num_cols = ["Age", "Tenure_Years", "Balance_INR", "Num_Products", "Has_Credit_Card",
                    "Is_Active_Member", "Estimated_Salary_INR", "Credit_Score",
                    "Balance_to_Salary", "Tenure_NumProducts", "Low_Credit_Score", "HighBalance_LowActivity"]
        cat_cols = ["Gender", "State", "Account_Type"]

        X_num = scaler.transform(input_df[num_cols])
        X_cat = encoder.transform(input_df[cat_cols])
        X_final = pd.DataFrame(np.concatenate([X_num, X_cat], axis=1), columns=feature_names)

        prob = model.predict_proba(X_final)[0][1]
        risk = "🔴 HIGH RISK" if prob >= 0.6 else ("🟡 MEDIUM RISK" if prob >= 0.3 else "🟢 LOW RISK")
        risk_color = "#F87171" if prob >= 0.6 else ("#FBBF24" if prob >= 0.3 else "#34D399")

        rc1, rc2 = st.columns([1, 2])
        with rc1:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;border-top:3px solid {risk_color};">
                <div style="font-size:2.5rem;font-weight:800;color:{risk_color};">{prob:.0%}</div>
                <div style="font-size:1rem;font-weight:700;color:{risk_color};margin:8px 0;">{risk}</div>
                <div style="font-size:.8rem;color:#94A3B8;">Churn Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100,
                number=dict(suffix="%", font=dict(size=36, color=risk_color)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#64748B"),
                    bar=dict(color=risk_color),
                    bgcolor="#1a1a1a",
                    steps=[
                        dict(range=[0, 30],  color="rgba(52,211,153,.15)"),
                        dict(range=[30, 60], color="rgba(251,191,36,.15)"),
                        dict(range=[60, 100], color="rgba(248,113,113,.15)"),
                    ],
                    threshold=dict(line=dict(color="white", width=2), thickness=0.8, value=prob * 100)
                )
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1"),
                margin=dict(t=30, b=10, l=30, r=30), height=250
            )
            st.plotly_chart(fig, use_container_width=True)

        if prob >= 0.6:
            st.error("⚠️ **Immediate intervention required** — Assign relationship manager and offer retention package.")
        elif prob >= 0.3:
            st.warning("📌 **Monitor closely** — Proactive engagement and personalized offers recommended.")
        else:
            st.success("✅ **Stable customer** — Focus on cross-selling and loyalty programs.")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — BUSINESS STRATEGY
# ══════════════════════════════════════════════════════════════
elif section == "📌 Business Strategy":
    st.markdown('<div class="section-header"><span>📌</span> Retention Strategy Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#CBD5E1;margin:0;">Translating ML predictions into <b style="color:#FBBF24;">actionable business strategies</b>
        across three risk tiers.</p>
    </div>
    """, unsafe_allow_html=True)

    strategies = [
        ("🟢", "Low Risk",    "< 30%",  "#34D399", "rgba(52,211,153,.08)",
         ["Cross-sell premium products (FDs, Mutual Funds)", "Loyalty reward programs",
          "Encourage digital banking adoption", "Promote auto-pay & recurring deposits"]),
        ("🟡", "Medium Risk", "30–60%", "#FBBF24", "rgba(251,191,36,.08)",
         ["Personalized offers (fee waivers, cashback)", "Proactive support check-in calls",
          "Targeted product nudges based on history", "Short-term retention incentives"]),
        ("🔴", "High Risk",   "> 60%",  "#F87171", "rgba(248,113,113,.08)",
         ["Immediate relationship manager intervention", "Exclusive fee waivers & rate benefits",
          "Custom win-back retention plans", "Priority grievance resolution & account audit"]),
    ]

    for icon, title, pct, color, bg, actions in strategies:
        st.markdown(f"""
        <div class="glass-card" style="border-left:3px solid {color};background:{bg};">
            <h3 style="color:{color} !important;margin-top:0;">{icon} {title} Customers
                <span style="font-size:.85rem;font-weight:400;color:#94A3B8;">({pct} churn probability)</span>
            </h3>
            <ul style="color:#CBD5E1;margin-bottom:0;line-height:1.8;">
                {''.join(f'<li>{a}</li>' for a in actions)}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Key business insights
    st.markdown('<div class="section-header"><span>💡</span> Key Business Insights</div>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    insights = [
        ("🔕 Dormancy Risk",
         "Inactive members with high balances are the #1 churn signal (45.6% feature importance).",
         "#F87171"),
        ("📦 Complexity Risk",
         "Customers with 3+ products and low credit scores show account overload patterns.",
         "#FBBF24"),
        ("🛡️ Retention Shield",
         "Active members have significantly lower churn — engagement is the best defense.",
         "#34D399"),
    ]
    for col, (title, desc, color) in zip([ic1, ic2, ic3], insights):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid {color};">
                <h4 style="color:{color} !important;margin-top:0;">{title}</h4>
                <p style="color:#CBD5E1;font-size:.88rem;margin-bottom:0;line-height:1.6;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <b style="color:#FBBF24;">Pooja Dhamale</b> · Python · XGBoost · Streamlit<br>
    <span style="font-size:.72rem;">End-to-End Machine Learning Project · 2026</span>
</div>
""", unsafe_allow_html=True)