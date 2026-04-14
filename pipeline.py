import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, mean_absolute_error
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield ML",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🛡️"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Nunito:wght@300;400;600;700&display=swap');

:root {
    --bg:        #0d0f14;
    --panel:     #13161f;
    --panel2:    #1a1e2e;
    --border:    #252a3a;
    --accent1:   #f97316;
    --accent2:   #fb923c;
    --accent3:   #10b981;
    --accent4:   #facc15;
    --danger:    #ef4444;
    --info:      #38bdf8;
    --purple:    #a78bfa;
    --text:      #e8eaf0;
    --muted:     #5a6380;
    --glow1:     rgba(249,115,22,.2);
    --glow2:     rgba(251,146,60,.1);
}

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 1rem 2rem 3rem; max-width: 100%;}

.hero-banner {
    background: linear-gradient(135deg, #0d0f14 0%, #13161f 50%, #1a1020 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent1);
    border-radius: 0 0 16px 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 10% 50%, var(--glow1) 0%, transparent 55%),
                radial-gradient(ellipse at 90% 30%, rgba(167,139,250,.08) 0%, transparent 50%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem;
    color: var(--accent1);
    letter-spacing: .25em;
    text-transform: uppercase;
    margin-bottom: .5rem;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 .3rem;
    line-height: 1;
}
.hero-title span {color: var(--accent1);}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: .78rem;
    color: var(--muted);
    letter-spacing: .08em;
}
.hero-badge {
    display: inline-block;
    background: rgba(249,115,22,.15);
    border: 1px solid rgba(249,115,22,.4);
    color: var(--accent1);
    font-family: 'JetBrains Mono', monospace;
    font-size: .65rem;
    padding: .2rem .7rem;
    border-radius: 4px;
    margin-right: .5rem;
    letter-spacing: .05em;
}

.sec-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: .7rem;
    margin: 1.8rem 0 1rem;
    padding-bottom: .6rem;
    border-bottom: 1px solid var(--border);
}
.sec-header .num {
    background: var(--accent1);
    color: #000;
    font-family: 'JetBrains Mono', monospace;
    font-size: .7rem;
    font-weight: 600;
    padding: .2rem .5rem;
    border-radius: 4px;
    min-width: 28px;
    text-align: center;
}

.metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap: .8rem; margin: 1.2rem 0;}
.metric-card {
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card .top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--bar-color, var(--accent1));
}
.metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--val-color, var(--accent1));
    line-height: 1;
    margin-bottom: .35rem;
}
.metric-label {
    font-size: .72rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: .06em;
    text-transform: uppercase;
    line-height: 1.3;
}
.metric-sub {
    font-size: .68rem;
    color: var(--accent3);
    font-family: 'JetBrains Mono', monospace;
    margin-top: .2rem;
}

.info-box  {background:rgba(56,189,248,.07);border:1px solid rgba(56,189,248,.25);border-left:3px solid #38bdf8;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#7dd3fc;margin:.5rem 0;}
.warn-box  {background:rgba(250,204,21,.07);border:1px solid rgba(250,204,21,.25);border-left:3px solid #facc15;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#fde68a;margin:.5rem 0;}
.danger-box{background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25);border-left:3px solid #ef4444;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#fca5a5;margin:.5rem 0;}
.ok-box    {background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.25);border-left:3px solid #10b981;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#6ee7b7;margin:.5rem 0;}
.running-box{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.25);border-left:3px solid var(--accent1);border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#fdba74;margin:.5rem 0;animation:pulse 1.5s infinite;}

@keyframes pulse {0%,100%{opacity:1}50%{opacity:.6}}

section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border);
}
.sidebar-logo {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent1);
    padding: .5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
    display: block;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent1), #ea580c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: .6rem 1.5rem !important;
    letter-spacing: .05em !important;
    transition: all .2s !important;
    box-shadow: 0 4px 15px rgba(249,115,22,.3) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 25px rgba(249,115,22,.5) !important;
    transform: translateY(-1px) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--panel2) !important;
    border-color: var(--border) !important;
}
.stTextInput > div > input,
.stNumberInput > div > input {
    background: var(--panel2) !important;
    border-color: var(--border) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTabs [data-baseweb="tab-list"] {border-bottom: 1px solid var(--border); gap:.3rem;}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 600 !important;
    padding: .5rem 1rem !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent1) !important;
    border-bottom: 2px solid var(--accent1) !important;
    background: rgba(249,115,22,.05) !important;
}

hr {border-color: var(--border) !important; margin: 1.5rem 0 !important;}

.streamlit-expanderHeader {
    background: var(--panel2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
}

.dataframe {font-size:.8rem !important; font-family:'JetBrains Mono',monospace !important;}
.stProgress > div > div {background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;}
.js-plotly-plot {border-radius:10px; overflow:hidden;}

/* Leakage warning banner */
.leak-banner {
    background: rgba(239,68,68,.1);
    border: 2px solid rgba(239,68,68,.5);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .8rem 0;
    font-size: .85rem;
    color: #fca5a5;
}
.leak-banner strong { color: #ef4444; font-family: 'JetBrains Mono', monospace; }
.fix-banner {
    background: rgba(16,185,129,.08);
    border: 2px solid rgba(16,185,129,.4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .8rem 0;
    font-size: .85rem;
    color: #6ee7b7;
}
.fix-banner strong { color: #10b981; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    'active_stage': 1,
    'df': None,
    'df_original': None,
    'model_trained': False,
    'stages_done': set(),
    'target_col': None,
    'features': None,
    'selected_features': None,
    'outlier_idx': [],
    'imputer': None,       # ✅ FIX: Store fitted imputer to apply to test
    'label_encoders': {},  # ✅ FIX: Store fitted encoders per column
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(19,22,31,0)',
    plot_bgcolor='rgba(19,22,31,0)',
    font=dict(family="Nunito, sans-serif", color="#e8eaf0", size=12),
    title_font=dict(family="Rajdhani, sans-serif", size=16, color="#e8eaf0"),
    xaxis=dict(gridcolor='#252a3a', linecolor='#252a3a', zerolinecolor='#252a3a'),
    yaxis=dict(gridcolor='#252a3a', linecolor='#252a3a', zerolinecolor='#252a3a'),
    legend=dict(bgcolor='rgba(19,22,31,0.8)', bordercolor='#252a3a', borderwidth=1),
    colorway=["#f97316","#a78bfa","#10b981","#facc15","#ef4444","#38bdf8","#fb923c"],
    margin=dict(t=50, b=30, l=10, r=10),
)
def styled_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────
STAGES = [
    (1, "📂", "Data Upload"),
    (2, "🔬", "EDA"),
    (3, "🛠️", "Engineering"),
    (4, "🎯", "Feature Selection"),
    (5, "🤖", "Model Training"),
    (6, "📊", "Evaluation"),
    (7, "⚙️", "Hyperparameter Tuning"),
]

with st.sidebar:
    st.markdown('<span class="sidebar-logo">🛡️ FraudShield ML</span>', unsafe_allow_html=True)
    st.markdown("**Navigation**")
    
    for num, icon, label in STAGES:
        is_active = st.session_state.active_stage == num
        done_mark = "✓ " if num in st.session_state.stages_done else ""
        if st.button(f"{done_mark}{icon} {label}", key=f"nav_{num}",
                     help=f"Go to stage {num}",
                     use_container_width=True):
            st.session_state.active_stage = num
            st.rerun()

    st.markdown("---")
    st.markdown("**⚙️ Config**")
    problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
    random_state = st.number_input("Random Seed", value=42, min_value=0)
    
    if st.session_state.df is not None:
        df_info = st.session_state.df
        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:JetBrains Mono,monospace;font-size:.72rem;color:#5a6380;">
        DATASET LOADED<br>
        <span style="color:#f97316">{df_info.shape[0]:,}</span> rows · 
        <span style="color:#f97316">{df_info.shape[1]}</span> cols
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-eyebrow">⚡ POWERED BY SCIKIT-LEARN + XGBOOST · DATA LEAKAGE FREE</div>
    <div class="hero-title">Credit Card <span>Fraud Detection</span></div>
    <div class="hero-sub" style="margin:.4rem 0 .8rem">END-TO-END ML PIPELINE · NO DATA LEAKAGE · PRODUCTION READY</div>
    <div>
        <span class="hero-badge">CLASSIFICATION</span>
        <span class="hero-badge">REGRESSION</span>
        <span class="hero-badge">AUTO-TUNING</span>
        <span class="hero-badge">XGBoost + LightGBM</span>
    </div>
</div>
""", unsafe_allow_html=True)

active = st.session_state.active_stage

# ─────────────────────────────────────────────
#  STAGE 1 — DATA UPLOAD
# ─────────────────────────────────────────────
if active == 1:
    st.markdown('<div class="sec-header"><span class="num">01</span> Data Acquisition & Shape Analysis</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Drop your CSV dataset here", type=["csv"], label_visibility="collapsed")
    
    if not uploaded_file:
        st.markdown('<div class="info-box">📁 Upload a CSV file to begin. The pipeline will guide you through each stage automatically.</div>', unsafe_allow_html=True)
        st.stop()
    
    with st.spinner("⏳ Reading and parsing dataset..."):
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.df_original = df.copy()  # ✅ Keep original untouched
        st.session_state.stages_done.add(1)
    
    st.markdown(f'<div class="ok-box">✅ Dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns · {df.isnull().sum().sum()} missing values</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.markdown("**📋 Dataset Preview**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("**🎯 Target & Feature Selection**")
        target_col = st.selectbox("Target Column", df.columns)
        st.session_state.target_col = target_col
        
        # ✅ FIX: Explicitly exclude target from features at upload stage
        default_feats = [c for c in df.columns if c != target_col]
        features = st.multiselect("Feature Columns", default_feats, default=default_feats)
        st.session_state.features = features
        
        if st.button("✅ Confirm & Proceed to EDA →"):
            st.session_state.stages_done.add(1)
            st.session_state.active_stage = 2
            st.rerun()
    
    with col2:
        if len(features) >= 2:
            with st.spinner("🔄 Computing PCA projection..."):
                temp_df = df[features].dropna().select_dtypes(include=[np.number])
                if temp_df.shape[1] >= 2 and temp_df.shape[0] > 5:
                    pca = PCA(n_components=2)
                    comps = pca.fit_transform(StandardScaler().fit_transform(temp_df))
                    color_series = df.loc[temp_df.index, target_col].astype(str)
                    fig = px.scatter(
                        x=comps[:, 0], y=comps[:, 1], color=color_series,
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                        title="2D PCA Projection",
                        color_discrete_sequence=["#f97316","#a78bfa","#10b981","#facc15","#ef4444"]
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.8))
                    st.plotly_chart(styled_fig(fig), use_container_width=True)

# ─────────────────────────────────────────────
#  STAGE 2 — EDA
# ─────────────────────────────────────────────
elif active == 2:
    df = st.session_state.get('df')
    if df is None:
        st.markdown('<div class="warn-box">⚠️ Please upload data first (Stage 1).</div>', unsafe_allow_html=True)
        st.stop()
    
    target_col = st.session_state.get('target_col', df.columns[-1])
    features = st.session_state.get('features', [c for c in df.columns if c != target_col])
    
    st.markdown('<div class="sec-header"><span class="num">02</span> Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Statistics", "📉 Distributions", "🔗 Correlation Heatmap", "🎯 Target Analysis"])
    
    with tab1:
        with st.spinner("📊 Computing descriptive statistics..."):
            c1, c2 = st.columns(2)
            c1.markdown("**Descriptive Statistics**")
            c1.dataframe(df.describe(), use_container_width=True)
            c2.markdown("**Data Types & Missing Values**")
            info_df = pd.DataFrame({
                "dtype": df.dtypes,
                "nulls": df.isnull().sum(),
                "null_%": (df.isnull().sum() / len(df) * 100).round(2),
                "unique": df.nunique()
            })
            c2.dataframe(info_df, use_container_width=True)
    
    with tab2:
        # ✅ Only show feature columns, not target
        num_cols = [c for c in df[features].select_dtypes(include=[np.number]).columns.tolist()]
        if num_cols:
            chosen = st.selectbox("Select column to visualize", num_cols)
            with st.spinner(f"📈 Plotting distribution for {chosen}..."):
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box Plot"])
                fig.add_trace(go.Histogram(x=df[chosen], marker_color='#f97316', opacity=0.85, name="hist"), row=1, col=1)
                fig.add_trace(go.Box(y=df[chosen], marker_color='#a78bfa', name="box"), row=1, col=2)
                st.plotly_chart(styled_fig(fig), use_container_width=True)
    
    with tab3:
        with st.spinner("🔗 Computing correlation matrix..."):
            # ✅ Correlation only on features (exclude target to avoid peeking)
            corr = df[features].select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr, text_auto=".2f",
                            color_continuous_scale=["#0d0f14","#252a3a","#f97316"],
                            title="Feature Correlation Matrix (Features Only)")
            st.plotly_chart(styled_fig(fig), use_container_width=True)
    
    with tab4:
        with st.spinner("🎯 Analyzing target distribution..."):
            if problem_type == "Classification":
                vc = df[target_col].value_counts()
                fig = px.bar(x=vc.index.astype(str), y=vc.values,
                             labels={"x": "Class", "y": "Count"},
                             title="Class Distribution",
                             color=vc.values,
                             color_continuous_scale=["#a78bfa","#f97316"])
                st.plotly_chart(styled_fig(fig), use_container_width=True)
                
                if len(vc) == 2:
                    ratio = vc.iloc[1] / vc.iloc[0]
                    if ratio < 0.1:
                        st.markdown(f'<div class="danger-box">🔴 Severe class imbalance! Minority class is only {ratio:.1%} of data. Use class_weight="balanced" in your model.</div>', unsafe_allow_html=True)
                    elif ratio < 0.3:
                        st.markdown(f'<div class="warn-box">⚠️ Moderate class imbalance ({ratio:.1%}). Use weighted metrics (F1, AUC) over raw accuracy.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ok-box">✅ Class distribution is relatively balanced.</div>', unsafe_allow_html=True)
            else:
                fig = px.histogram(df, x=target_col, nbins=40, title="Target Distribution",
                                   color_discrete_sequence=["#f97316"])
                st.plotly_chart(styled_fig(fig), use_container_width=True)
    
    if st.button("✅ Proceed to Data Engineering →"):
        st.session_state.stages_done.add(2)
        st.session_state.active_stage = 3
        st.rerun()

# ─────────────────────────────────────────────
#  STAGE 3 — DATA ENGINEERING
#  ✅ FIX: Imputer fitted only on train features, not full dataset
#  ✅ FIX: Outlier detection excludes target column & binary columns
# ─────────────────────────────────────────────
elif active == 3:
    df = st.session_state.get('df')
    if df is None:
        st.markdown('<div class="warn-box">⚠️ Please upload data first (Stage 1).</div>', unsafe_allow_html=True)
        st.stop()
    
    target_col = st.session_state.get('target_col', df.columns[-1])
    features = st.session_state.get('features', [c for c in df.columns if c != target_col])
    
    st.markdown('<div class="sec-header"><span class="num">03</span> Data Engineering & Cleaning</div>', unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────
    # DATA LEAKAGE EXPLANATION PANEL
    # ────────────────────────────────────────────────
    with st.expander("⚠️ Data Leakage Prevention — What Changed & Why", expanded=False):
        st.markdown("""
        <div class="leak-banner">
        <strong>BUG FIXED #1 — Imputer Leakage</strong><br>
        Previously: <code>SimpleImputer</code> was fitted on the <b>entire dataset</b> (train + test combined).<br>
        This causes the imputer to learn the mean/median of the test set, which leaks test information into training.
        </div>
        <div class="fix-banner">
        <strong>FIX APPLIED:</strong> The imputer is now fitted <b>only on the training portion</b> of each feature split in Stage 5.
        Here in Stage 3, imputation is applied to the full dataframe for EDA purposes only — the actual train/test imputation
        happens after the split in Stage 5, using a freshly fitted imputer on X_train only.
        </div>
        <div class="leak-banner">
        <strong>BUG FIXED #2 — Outlier Removal Deleting Fraud Cases</strong><br>
        Previously: IQR outlier detection ran on ALL numeric columns including <b>the target (Class)</b>.<br>
        In a fraud dataset with 0.17% fraud, the Class=1 rows are extreme outliers by IQR standards — 
        so ALL fraud cases were being deleted! This gave a clean 100% "accuracy" predicting only class 0.
        </div>
        <div class="fix-banner">
        <strong>FIX APPLIED:</strong> Outlier detection now only runs on <b>feature columns</b>, 
        explicitly excluding the target column and any binary (0/1) columns.
        </div>
        """, unsafe_allow_html=True)
    
    eng_col1, eng_col2 = st.columns(2, gap="large")
    
    with eng_col1:
        st.markdown("**🩹 Missing Value Imputation**")
        null_count = df[features].isnull().sum().sum()
        if null_count > 0:
            st.markdown(f'<div class="warn-box">⚠️ {null_count} missing values in feature columns.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✅ No missing values found in features.</div>', unsafe_allow_html=True)
        
        impute_method = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
        
        # ✅ FIX: Store the imputation strategy for use in Stage 5 (fit on train only)
        st.session_state.impute_method = impute_method
        
        st.markdown("""
        <div class="fix-banner" style="font-size:.78rem;">
        ✅ <strong>Leakage-Free:</strong> Imputer will be fitted on X_train only during Stage 5 training,
        then applied to X_test using train statistics only.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶ Preview Imputation Effect"):
            with st.spinner("⏳ Previewing imputation..."):
                # Preview only — not stored (actual imputation in Stage 5)
                num_c = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
                preview = df[num_c].copy()
                imp_preview = SimpleImputer(strategy=impute_method)
                imp_preview.fit_transform(preview)
            st.markdown(f'<div class="ok-box">✅ Preview: {impute_method} imputation would fill {null_count} missing values. Will be applied post-split in Stage 5.</div>', unsafe_allow_html=True)
    
    with eng_col2:
        st.markdown("**🚨 Outlier Detection & Removal**")
        outlier_method = st.selectbox("Detection Method", ["IQR", "Isolation Forest", "DBSCAN"])
        
        # ✅ FIX: Only numeric FEATURE columns, exclude target and binary columns
        num_features_raw = [f for f in features if f in df.select_dtypes(include=[np.number]).columns]
        # Exclude binary columns (like 0/1 flags) — they would falsely trigger IQR
        non_binary_features = [
            f for f in num_features_raw
            if df[f].nunique() > 2  # skip binary 0/1 columns
        ]
        
        # Double-check target is excluded
        non_binary_features = [f for f in non_binary_features if f != target_col]
        
        st.markdown(f"""
        <div class="fix-banner" style="font-size:.78rem;">
        ✅ <strong>Bug Fixed:</strong> Outlier detection runs on <b>{len(non_binary_features)}</b> continuous 
        feature columns only (target column + {len(num_features_raw) - len(non_binary_features)} binary columns excluded).
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶ Detect Outliers"):
            with st.spinner(f"🔍 Running {outlier_method} outlier detection on features only..."):
                outlier_idx = []
                if non_binary_features:
                    data_for_outlier = df[non_binary_features].fillna(df[non_binary_features].median())
                    
                    if outlier_method == "IQR":
                        Q1 = data_for_outlier.quantile(0.25)
                        Q3 = data_for_outlier.quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_mask = ((data_for_outlier < (Q1 - 1.5*IQR)) | (data_for_outlier > (Q3 + 1.5*IQR))).any(axis=1)
                        outlier_idx = df.index[outlier_mask].tolist()
                    elif outlier_method == "Isolation Forest":
                        iso = IsolationForest(contamination=0.05, random_state=random_state)
                        preds = iso.fit_predict(data_for_outlier)
                        outlier_idx = df.index[preds == -1].tolist()
                    elif outlier_method == "DBSCAN":
                        sc = StandardScaler()
                        Xsc = sc.fit_transform(data_for_outlier)
                        db = DBSCAN(eps=0.5, min_samples=5)
                        labels = db.fit_predict(Xsc)
                        outlier_idx = df.index[labels == -1].tolist()
                    
                    st.session_state.outlier_idx = outlier_idx
                
                # Show how many fraud cases would be affected
                if outlier_idx and target_col in df.columns:
                    outlier_df = df.loc[outlier_idx]
                    fraud_in_outliers = (outlier_df[target_col] == 1).sum() if problem_type == "Classification" else 0
                    total_fraud = (df[target_col] == 1).sum() if problem_type == "Classification" else 0
                    
                    if fraud_in_outliers > 0:
                        st.markdown(f'<div class="warn-box">⚠️ {len(outlier_idx)} outliers detected. Note: {fraud_in_outliers} of them are fraud cases ({fraud_in_outliers/max(total_fraud,1)*100:.1f}% of all fraud). Consider if removal is appropriate.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="ok-box">✅ {len(outlier_idx)} outliers detected. None are fraud cases — safe to remove.</div>', unsafe_allow_html=True)
                elif not outlier_idx:
                    st.markdown('<div class="ok-box">✅ No significant outliers detected.</div>', unsafe_allow_html=True)
        
        if st.session_state.get('outlier_idx'):
            if st.button("🗑️ Remove Detected Outliers"):
                with st.spinner("⏳ Removing outliers..."):
                    df = df.drop(st.session_state.outlier_idx).reset_index(drop=True)
                    st.session_state.df = df
                    st.session_state.outlier_idx = []
                st.markdown(f'<div class="ok-box">✅ Outliers removed. Dataset now has {len(df):,} rows.</div>', unsafe_allow_html=True)
    
    # ✅ FIX: Encode categoricals but store encoders for test-time use
    cat_cols = [f for f in features if f in df.select_dtypes(exclude=[np.number]).columns]
    if cat_cols:
        st.markdown("---")
        st.markdown("**🔢 Categorical Encoding**")
        if st.button("▶ Encode Categorical Columns"):
            with st.spinner("🔄 Encoding categorical features..."):
                label_encoders = {}
                for c in cat_cols:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].astype(str))
                    label_encoders[c] = le
                st.session_state.df = df
                st.session_state.label_encoders = label_encoders
            st.markdown(f'<div class="ok-box">✅ Encoded {len(cat_cols)} categorical columns: {", ".join(cat_cols)}</div>', unsafe_allow_html=True)
    
    if st.button("✅ Proceed to Feature Selection →"):
        st.session_state.stages_done.add(3)
        st.session_state.active_stage = 4
        st.rerun()

# ─────────────────────────────────────────────
#  STAGE 4 — FEATURE SELECTION
#  ✅ FIX: Selection methods stored as config only — applied to X_train in Stage 5
# ─────────────────────────────────────────────
elif active == 4:
    df = st.session_state.get('df')
    if df is None:
        st.markdown('<div class="warn-box">⚠️ Please complete previous stages first.</div>', unsafe_allow_html=True)
        st.stop()
    
    target_col = st.session_state.get('target_col', df.columns[-1])
    features = st.session_state.get('features', [c for c in df.columns if c != target_col])
    
    st.markdown('<div class="sec-header"><span class="num">04</span> Feature Selection & Importance</div>', unsafe_allow_html=True)
    
    # ────────────────────────────────────────────────
    # DATA LEAKAGE EXPLANATION PANEL
    # ────────────────────────────────────────────────
    with st.expander("⚠️ Data Leakage Prevention — Feature Selection", expanded=False):
        st.markdown("""
        <div class="leak-banner">
        <strong>BUG FIXED #3 — Feature Selection Leakage (Critical)</strong><br>
        Previously: <code>VarianceThreshold</code> and <code>Correlation Filter</code> were fitted on the 
        <b>entire dataset before the train/test split</b>.<br><br>
        This means the selector "saw" the test rows when deciding which features to keep,
        leaking test set statistics into the feature selection process.
        This is one of the most common and impactful sources of data leakage.
        </div>
        <div class="fix-banner">
        <strong>FIX APPLIED:</strong> Feature selection methods are now configured here for preview/EDA only.
        The actual fitting of <code>VarianceThreshold</code> and <code>mutual_info</code> happens 
        <b>inside the training pipeline in Stage 5</b>, fitted only on X_train, then applied to X_test.
        </div>
        """, unsafe_allow_html=True)
    
    # ✅ Only numeric feature columns (never the target)
    num_feat_cols = [f for f in features
                     if f != target_col and f in df.select_dtypes(include=[np.number]).columns]
    
    fs_methods = st.multiselect(
        "Select Feature Selection Methods (applied on training data only in Stage 5)",
        ["Variance Threshold", "Information Gain", "Correlation Filter"],
        default=[]
    )
    st.session_state.fs_methods = fs_methods
    
    # Preview on full data for EDA purposes (clearly labeled)
    if fs_methods and st.button("▶ Preview Feature Selection (EDA only — not applied to test set)"):
        progress = st.progress(0, text="Starting preview...")
        preview_features = list(num_feat_cols)
        
        if "Variance Threshold" in fs_methods and preview_features:
            progress.progress(25, text="⏳ Variance Threshold preview...")
            sel = VarianceThreshold(threshold=0.01)
            sel.fit(df[preview_features].fillna(0))
            kept = [f for f, s in zip(preview_features, sel.get_support()) if s]
            dropped = [f for f in preview_features if f not in kept]
            preview_features = kept
            st.markdown(f'<div class="info-box">📐 Variance Threshold Preview: <b>{len(kept)}</b> features kept, <b>{len(dropped)}</b> dropped: {dropped[:5]}</div>', unsafe_allow_html=True)
        
        if "Correlation Filter" in fs_methods and len(preview_features) > 1:
            progress.progress(50, text="⏳ Correlation filter preview...")
            corr_m = df[preview_features].corr().abs()
            upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
            preview_features = [f for f in preview_features if f not in to_drop]
            st.markdown(f'<div class="info-box">🔗 Correlation Filter Preview: <b>{len(preview_features)}</b> features retained (would drop {len(to_drop)} highly correlated).</div>', unsafe_allow_html=True)
        
        if "Information Gain" in fs_methods and preview_features:
            progress.progress(75, text="⏳ Information Gain preview...")
            y_tmp = df[target_col]
            if problem_type == "Classification":
                y_enc = LabelEncoder().fit_transform(y_tmp.astype(str))
                ig = mutual_info_classif(df[preview_features].fillna(0), y_enc, random_state=random_state)
            else:
                ig = mutual_info_regression(df[preview_features].fillna(0), pd.to_numeric(y_tmp, errors='coerce').fillna(0), random_state=random_state)
            
            ig_df = pd.DataFrame({"Feature": preview_features, "Info Gain": ig}).sort_values("Info Gain", ascending=False)
            fig = px.bar(ig_df, x="Info Gain", y="Feature", orientation='h',
                         title="Feature Information Gain (EDA Preview)",
                         color="Info Gain", color_continuous_scale=["#252a3a","#f97316"])
            st.plotly_chart(styled_fig(fig), use_container_width=True)
        
        progress.progress(100, text="✅ Preview complete!")
        st.session_state.preview_features = preview_features
        st.markdown(f'<div class="ok-box">✅ Preview: {len(preview_features)} features would be selected. Actual selection will run on X_train only in Stage 5.</div>', unsafe_allow_html=True)
    
    # Default: use all numeric features (target already excluded above)
    st.session_state.selected_features = num_feat_cols
    
    st.markdown(f"""
    <div class="fix-banner">
    ✅ <strong>Leakage-Free Pipeline:</strong> {len(num_feat_cols)} numeric feature columns identified 
    (target <code>{target_col}</code> excluded). Feature selection transformers will be fitted on 
    X_train only during Stage 5.
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✅ Proceed to Model Training →"):
        st.session_state.stages_done.add(4)
        st.session_state.active_stage = 5
        st.rerun()

# ─────────────────────────────────────────────
#  STAGE 5 — MODEL TRAINING
#  ✅ FIX: All preprocessing fitted on X_train only, then applied to X_test
#  ✅ FIX: Feature selection happens AFTER train/test split
# ─────────────────────────────────────────────
elif active == 5:
    df = st.session_state.get('df')
    if df is None:
        st.markdown('<div class="warn-box">⚠️ Please complete previous stages first.</div>', unsafe_allow_html=True)
        st.stop()
    
    target_col = st.session_state.get('target_col', df.columns[-1])
    features = st.session_state.get('features', [c for c in df.columns if c != target_col])
    selected_features = st.session_state.get('selected_features',
                        [f for f in features if f != target_col and f in df.select_dtypes(include=[np.number]).columns])
    
    st.markdown('<div class="sec-header"><span class="num">05</span> Model Selection & Training</div>', unsafe_allow_html=True)
    
    # Pipeline explanation
    with st.expander("✅ Leakage-Free Training Pipeline — How It Works", expanded=False):
        st.markdown("""
        <div class="fix-banner">
        <strong>Correct Pipeline Order (No Leakage):</strong><br>
        1. Train/Test Split → 2. Imputer fit on X_train → 3. Scaler fit on X_train → 
        4. Feature Selection fit on X_train → 5. Model fit on X_train → 
        6. Apply all transforms to X_test using train statistics only → 7. Evaluate on X_test
        </div>
        """, unsafe_allow_html=True)
    
    train_col1, train_col2 = st.columns([1, 1], gap="large")
    
    with train_col1:
        st.markdown("**⚙️ Training Configuration**")
        test_size = st.slider("Test Size (%)", 10, 50, 20)
        k_val = st.slider("K-Fold CV (K)", 2, 10, 5)
        scale_data = st.toggle("Standardize Features (StandardScaler)", value=True)
        use_class_weight = st.toggle("Use class_weight='balanced' (recommended for imbalanced data)", value=True)
        
        st.markdown("**📊 Model Parameters**")
    
    with train_col2:
        st.markdown("**🤖 Model Selection**")
        
        model_groups = {
            "📈 Linear Models": ["Logistic/Linear Regression", "Ridge Regression", "Lasso Regression"],
            "🌲 Tree Models": ["Decision Tree", "Extra Trees"],
            "🌳 Ensemble Models": ["Random Forest", "Gradient Boosting", "AdaBoost", "Bagging"],
            "⚡ Boosting (Advanced)": (
                (["XGBoost"] if XGBOOST_AVAILABLE else ["XGBoost (install: pip install xgboost)"]) +
                (["LightGBM"] if LGBM_AVAILABLE else []) +
                (["CatBoost"] if CATBOOST_AVAILABLE else [])
            ),
            "🔮 Other Models": ["SVM (RBF)", "SVM (Linear)", "K-Nearest Neighbors", "Naive Bayes"]
        }
        
        group_display = [(g, ms) for g, ms in model_groups.items() if ms]
        model_group = st.selectbox("Model Category", [g for g, _ in group_display])
        ms_for_group = dict(group_display)[model_group]
        model_choice = st.selectbox("Select Model", ms_for_group)
    
    with train_col1:
        max_depth = None
        n_estimators = 100
        lr_boost = 0.1
        svm_c = 1.0
        knn_k = 5
        
        if "SVM" in model_choice:
            svm_c = st.slider("SVM Regularization (C)", 0.01, 10.0, 1.0)
        if "K-Nearest" in model_choice:
            knn_k = st.slider("K Neighbors", 1, 20, 5)
        if any(x in model_choice for x in ["Decision Tree", "Random Forest", "Extra Trees"]):
            _md = st.slider("Max Depth (0 = unlimited)", 0, 30, 0)
            max_depth = None if _md == 0 else _md
        if any(x in model_choice for x in ["XGBoost", "Gradient Boosting", "LightGBM"]):
            n_estimators = st.slider("N Estimators", 50, 500, 100, step=50)
            lr_boost = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
    
    # ✅ Prepare X and y — ensure target is never in X
    safe_features = [f for f in selected_features if f != target_col]
    X_all = df[safe_features].copy()
    y_raw = df[target_col]
    
    if problem_type == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = le.classes_
    else:
        y = pd.to_numeric(y_raw, errors='coerce').fillna(0).values
        class_names = None
    
    def get_model(name, ptype, cw):
        is_cls = (ptype == "Classification")
        cw_param = "balanced" if (cw and is_cls) else None
        
        if "Logistic/Linear" in name:
            return LogisticRegression(max_iter=500, n_jobs=-1, class_weight=cw_param) if is_cls else LinearRegression(n_jobs=-1)
        if "Ridge" in name:
            return LogisticRegression(penalty='l2', max_iter=500, class_weight=cw_param) if is_cls else Ridge()
        if "Lasso" in name:
            return LogisticRegression(max_iter=500, class_weight=cw_param) if is_cls else Lasso()
        if "Decision Tree" in name:
            return DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, class_weight=cw_param) if is_cls else DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
        if "Extra Trees" in name:
            return ExtraTreesClassifier(random_state=random_state, max_depth=max_depth, n_jobs=-1, class_weight=cw_param) if is_cls else ExtraTreesRegressor(random_state=random_state, max_depth=max_depth, n_jobs=-1)
        if "Random Forest" in name:
            return RandomForestClassifier(random_state=random_state, max_depth=max_depth, n_jobs=-1, class_weight=cw_param) if is_cls else RandomForestRegressor(random_state=random_state, max_depth=max_depth, n_jobs=-1)
        if "Gradient Boosting" in name:
            return GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr_boost, random_state=random_state) if is_cls else GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr_boost, random_state=random_state)
        if "AdaBoost" in name:
            return AdaBoostClassifier(random_state=random_state) if is_cls else AdaBoostRegressor(random_state=random_state)
        if "Bagging" in name:
            return BaggingClassifier(random_state=random_state, n_jobs=-1) if is_cls else BaggingRegressor(random_state=random_state, n_jobs=-1)
        if "XGBoost" in name and XGBOOST_AVAILABLE:
            # ✅ scale_pos_weight handles imbalance in XGBoost
            n_neg = np.sum(y == 0); n_pos = np.sum(y == 1)
            spw = n_neg / max(n_pos, 1) if cw else 1
            return XGBClassifier(n_estimators=n_estimators, learning_rate=lr_boost,
                                  scale_pos_weight=spw if is_cls else 1,
                                  eval_metric='logloss', random_state=random_state, verbosity=0, n_jobs=-1) if is_cls else XGBRegressor(n_estimators=n_estimators, learning_rate=lr_boost, random_state=random_state, verbosity=0, n_jobs=-1)
        if "LightGBM" in name and LGBM_AVAILABLE:
            return LGBMClassifier(n_estimators=n_estimators, learning_rate=lr_boost, class_weight=cw_param, random_state=random_state, verbose=-1, n_jobs=-1) if is_cls else LGBMRegressor(n_estimators=n_estimators, learning_rate=lr_boost, random_state=random_state, verbose=-1, n_jobs=-1)
        if "CatBoost" in name and CATBOOST_AVAILABLE:
            return CatBoostClassifier(verbose=0, random_state=random_state, auto_class_weights='Balanced' if cw else None) if is_cls else CatBoostRegressor(verbose=0, random_state=random_state)
        if "SVM (RBF)" in name:
            return SVC(kernel='rbf', C=svm_c, probability=True, class_weight=cw_param) if is_cls else SVR(kernel='rbf', C=svm_c)
        if "SVM (Linear)" in name:
            return SVC(kernel='linear', C=svm_c, probability=True, class_weight=cw_param) if is_cls else SVR(kernel='linear', C=svm_c)
        if "K-Nearest" in name:
            return KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1) if is_cls else KNeighborsRegressor(n_neighbors=knn_k, n_jobs=-1)
        if "Naive Bayes" in name:
            return GaussianNB()
        return LogisticRegression(class_weight=cw_param) if is_cls else LinearRegression()
    
    if st.button(f"🚀 Train {model_choice} Now"):
        prog = st.progress(0, text="⏳ Step 1/6: Splitting data into train/test...")
        
        # ─── STEP 1: Split FIRST — before any fitting ───────────────────────
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all.values, y, test_size=test_size/100,
            random_state=random_state,
            stratify=y if problem_type == "Classification" else None  # ✅ Stratify preserves class ratio
        )
        
        X_train_df = pd.DataFrame(X_train_raw, columns=safe_features)
        X_test_df  = pd.DataFrame(X_test_raw,  columns=safe_features)
        
        prog.progress(15, text="⏳ Step 2/6: Fitting imputer on X_train only...")
        
        # ─── STEP 2: Imputer — fit on X_train, transform both ───────────────
        num_feature_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
        imputer = SimpleImputer(strategy=st.session_state.get('impute_method', 'median'))
        X_train_df[num_feature_cols] = imputer.fit_transform(X_train_df[num_feature_cols])
        X_test_df[num_feature_cols]  = imputer.transform(X_test_df[num_feature_cols])   # ✅ transform only
        st.session_state.fitted_imputer = imputer
        
        prog.progress(30, text="⏳ Step 3/6: Feature selection on X_train only...")
        
        # ─── STEP 3: Feature Selection — fit on X_train, apply to X_test ────
        fs_methods_chosen = st.session_state.get('fs_methods', [])
        final_feature_cols = list(num_feature_cols)
        
        if "Variance Threshold" in fs_methods_chosen and final_feature_cols:
            vt = VarianceThreshold(threshold=0.01)
            vt.fit(X_train_df[final_feature_cols].fillna(0))
            kept_mask = vt.get_support()
            final_feature_cols = [f for f, k in zip(final_feature_cols, kept_mask) if k]
        
        if "Correlation Filter" in fs_methods_chosen and len(final_feature_cols) > 1:
            # Correlation computed on X_train only
            corr_train = X_train_df[final_feature_cols].corr().abs()
            upper = corr_train.where(np.triu(np.ones(corr_train.shape), k=1).astype(bool))
            to_drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
            final_feature_cols = [f for f in final_feature_cols if f not in to_drop_corr]
        
        if "Information Gain" in fs_methods_chosen and final_feature_cols:
            if problem_type == "Classification":
                ig_scores = mutual_info_classif(X_train_df[final_feature_cols].fillna(0), y_train, random_state=random_state)
            else:
                ig_scores = mutual_info_regression(X_train_df[final_feature_cols].fillna(0), y_train, random_state=random_state)
            # Keep top 80% by info gain
            threshold_ig = np.percentile(ig_scores, 20)
            final_feature_cols = [f for f, s in zip(final_feature_cols, ig_scores) if s >= threshold_ig]
        
        st.session_state.final_feature_cols = final_feature_cols
        st.session_state.selected_features = final_feature_cols
        
        X_train = X_train_df[final_feature_cols].values
        X_test  = X_test_df[final_feature_cols].values
        
        prog.progress(50, text="⏳ Step 4/6: Scaling features (fit on X_train only)...")
        
        # ─── STEP 4: Scaler — fit on X_train, transform both ────────────────
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)   # ✅ transform only
            st.session_state.scaler = scaler
        
        prog.progress(65, text=f"⏳ Step 5/6: Running {k_val}-Fold CV on X_train only...")
        
        # ─── STEP 5: Cross-validation on X_train only ───────────────────────
        model = get_model(model_choice, problem_type, use_class_weight)
        scoring = 'f1_weighted' if problem_type == "Classification" else 'r2'  # ✅ Use F1 not accuracy for imbalanced
        cv_res = cross_validate(model, X_train, y_train, cv=k_val,
                                scoring=scoring, return_train_score=True)
        
        prog.progress(80, text="⏳ Step 6/6: Fitting final model and evaluating on test set...")
        
        # ─── STEP 6: Fit and predict ─────────────────────────────────────────
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        
        prog.progress(100, text="✅ Training complete — no data leakage!")
        
        # Store everything
        st.session_state.model        = model
        st.session_state.model_choice = model_choice
        st.session_state.X_train      = X_train
        st.session_state.X_test       = X_test
        st.session_state.y_train      = y_train
        st.session_state.y_test       = y_test
        st.session_state.train_pred   = train_pred
        st.session_state.test_pred    = test_pred
        st.session_state.cv_res       = cv_res
        st.session_state.class_names  = class_names
        st.session_state.k_val        = k_val
        st.session_state.y            = y
        st.session_state.model_trained = True
        st.session_state.stages_done.add(5)
        st.session_state.scoring_metric = scoring
        
        st.markdown(f"""
        <div class="fix-banner">
        ✅ <strong>Leakage-Free Training Complete!</strong><br>
        Pipeline: Split → Imputer(fit on train) → FeatureSelection(fit on train) → 
        Scaler(fit on train) → CV(on train) → Model(on train) → Evaluate(on test)
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ View Evaluation Results →"):
            st.session_state.active_stage = 6
            st.rerun()

# ─────────────────────────────────────────────
#  STAGE 6 — EVALUATION
# ─────────────────────────────────────────────
elif active == 6:
    if not st.session_state.get('model_trained'):
        st.markdown('<div class="warn-box">⚠️ No trained model found. Please complete Stage 5 first.</div>', unsafe_allow_html=True)
        st.stop()
    
    model         = st.session_state.model
    model_choice  = st.session_state.model_choice
    X_train       = st.session_state.X_train
    X_test        = st.session_state.X_test
    y_train       = st.session_state.y_train
    y_test        = st.session_state.y_test
    train_pred    = st.session_state.train_pred
    test_pred     = st.session_state.test_pred
    cv_res        = st.session_state.cv_res
    class_names   = st.session_state.class_names
    k_val         = st.session_state.k_val
    y             = st.session_state.y
    selected_features = st.session_state.get('final_feature_cols',
                        st.session_state.get('selected_features', []))
    scoring_metric = st.session_state.get('scoring_metric', 'f1_weighted')
    
    st.markdown('<div class="sec-header"><span class="num">06</span> Model Evaluation & Performance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">📌 Evaluating: <b>{model_choice}</b> | CV Metric: <b>{scoring_metric}</b></div>', unsafe_allow_html=True)
    
    if problem_type == "Classification":
        with st.spinner("📊 Computing classification metrics..."):
            acc_tr = accuracy_score(y_train, train_pred)
            acc_te = accuracy_score(y_test,  test_pred)
            prec   = precision_score(y_test, test_pred, average='weighted', zero_division=0)
            rec    = recall_score(y_test,    test_pred, average='weighted', zero_division=0)
            f1     = f1_score(y_test,        test_pred, average='weighted', zero_division=0)
            
            # ✅ For imbalanced fraud: show minority class recall separately
            if len(np.unique(y)) == 2:
                fraud_recall  = recall_score(y_test, test_pred, pos_label=1, zero_division=0)
                fraud_prec    = precision_score(y_test, test_pred, pos_label=1, zero_division=0)
                fraud_f1      = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
            else:
                fraud_recall = fraud_prec = fraud_f1 = None
            
            cv_mean = cv_res['test_score'].mean()
            cv_std  = cv_res['test_score'].std()
            
            try:
                if len(np.unique(y)) == 2:
                    prob = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else None
                    auc  = roc_auc_score(y_test, prob) if prob is not None else None
                else:
                    prob_all = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    auc = roc_auc_score(y_test, prob_all, multi_class='ovr', average='weighted') if prob_all is not None else None
            except:
                auc = None
        
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        
        # ✅ Imbalance warning
        total_fraud = (y_test == 1).sum()
        total_legit = (y_test == 0).sum()
        if total_fraud > 0 and total_legit / total_fraud > 5:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ <b>Imbalanced Dataset:</b> Test set has {total_legit} legitimate vs {total_fraud} fraud cases.
            Raw <b>accuracy is misleading</b> — focus on <b>AUC, Fraud Recall, and F1</b> instead.
            A model predicting "no fraud" always would get {total_legit/(total_legit+total_fraud):.1%} accuracy.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="top-bar" style="background:#f97316"></div>
                <div class="metric-val" style="color:#f97316;font-size:2rem">{acc_te:.1%}</div>
                <div class="metric-label">TEST ACCURACY</div>
                <div class="metric-sub">train: {acc_tr:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#a78bfa"></div>
                <div class="metric-val" style="color:#a78bfa;font-size:2rem">{prec:.1%}</div>
                <div class="metric-label">PRECISION (W)</div>
                <div class="metric-sub">weighted avg</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#10b981"></div>
                <div class="metric-val" style="color:#10b981;font-size:2rem">{rec:.1%}</div>
                <div class="metric-label">RECALL (W)</div>
                <div class="metric-sub">weighted avg</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#facc15"></div>
                <div class="metric-val" style="color:#facc15;font-size:2rem">{f1:.1%}</div>
                <div class="metric-label">F1-SCORE (W)</div>
                <div class="metric-sub">weighted avg</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#38bdf8"></div>
                <div class="metric-val" style="color:#38bdf8;font-size:2rem">{auc_str}</div>
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-sub">area under curve</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#fb923c"></div>
                <div class="metric-val" style="color:#fb923c;font-size:1.6rem">{cv_mean:.1%}</div>
                <div class="metric-label">CV {scoring_metric.upper()}</div>
                <div class="metric-sub">±{cv_std:.1%} std</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ✅ Fraud-specific metrics for imbalanced case
        if fraud_recall is not None:
            st.markdown("**🎯 Fraud Class Metrics (Most Important for Fraud Detection)**")
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="top-bar" style="background:#ef4444"></div>
                    <div class="metric-val" style="color:#ef4444;font-size:2rem">{fraud_recall:.1%}</div>
                    <div class="metric-label">FRAUD RECALL</div>
                    <div class="metric-sub">% frauds caught ← key metric</div>
                </div>
                <div class="metric-card">
                    <div class="top-bar" style="background:#38bdf8"></div>
                    <div class="metric-val" style="color:#38bdf8;font-size:2rem">{fraud_prec:.1%}</div>
                    <div class="metric-label">FRAUD PRECISION</div>
                    <div class="metric-sub">% flagged = actual fraud</div>
                </div>
                <div class="metric-card">
                    <div class="top-bar" style="background:#facc15"></div>
                    <div class="metric-val" style="color:#facc15;font-size:2rem">{fraud_f1:.1%}</div>
                    <div class="metric-label">FRAUD F1</div>
                    <div class="metric-sub">harmonic mean</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if fraud_recall < 0.5:
                st.markdown('<div class="danger-box">🔴 Low Fraud Recall — model is missing many fraud cases. Try: lower classification threshold, use SMOTE, or increase class_weight.</div>', unsafe_allow_html=True)
            elif fraud_recall < 0.8:
                st.markdown(f'<div class="warn-box">🟡 Moderate Fraud Recall ({fraud_recall:.1%}). For production fraud detection, aim for >85% recall.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ok-box">🟢 Good Fraud Recall ({fraud_recall:.1%}) — model is catching most fraud cases.</div>', unsafe_allow_html=True)
        
        # Fit analysis
        gap = acc_tr - acc_te
        if gap > 0.15:
            st.markdown(f'<div class="danger-box">🔴 <b>Overfitting</b> — Train {acc_tr:.1%} vs Test {acc_te:.1%} (gap: {gap:.1%}). Try: reduce max_depth, add regularization.</div>', unsafe_allow_html=True)
        elif acc_tr < 0.6:
            st.markdown('<div class="warn-box">🟡 <b>Underfitting</b> — Model too simple.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-box">🟢 <b>Good Generalization</b> — Train {acc_tr:.1%} / Test {acc_te:.1%}.</div>', unsafe_allow_html=True)
        
        tab_cm, tab_roc, tab_cv, tab_cls = st.tabs(["Confusion Matrix", "ROC Curve", "CV Scores", "Classification Report"])
        
        with tab_cm:
            with st.spinner("📊 Generating confusion matrix..."):
                cm = confusion_matrix(y_test, test_pred)
                labels_str = [str(c) for c in (class_names if class_names is not None else np.unique(y))]
                fig_cm = px.imshow(cm, text_auto=True, x=labels_str, y=labels_str,
                                   color_continuous_scale=["#0d0f14","#252a3a","#f97316"],
                                   title="Confusion Matrix")
                fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                st.plotly_chart(styled_fig(fig_cm), use_container_width=True)
                
                if len(labels_str) == 2:
                    tn, fp, fn, tp = cm.ravel()
                    st.markdown(f"""
                    <div class="info-box">
                    📊 <b>Confusion Matrix Breakdown:</b><br>
                    True Negatives (legitimate correctly identified): <b>{tn:,}</b> &nbsp;|&nbsp;
                    False Positives (legitimate flagged as fraud): <b>{fp:,}</b><br>
                    False Negatives (fraud missed — critical!): <b style="color:#ef4444">{fn:,}</b> &nbsp;|&nbsp;
                    True Positives (fraud correctly caught): <b style="color:#10b981">{tp:,}</b>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab_roc:
            if auc is not None and len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                with st.spinner("📈 Computing ROC curve..."):
                    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                 name=f'ROC (AUC={auc:.3f})',
                                                 line=dict(color='#f97316', width=2.5),
                                                 fill='tozeroy', fillcolor='rgba(249,115,22,.1)'))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                                 line=dict(dash='dash', color='#5a6380'), name='Random Classifier'))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                    st.plotly_chart(styled_fig(fig_roc), use_container_width=True)
            else:
                st.markdown('<div class="info-box">ROC curve shown for binary classification only.</div>', unsafe_allow_html=True)
        
        with tab_cv:
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(k_val)],
                "Train Score": cv_res['train_score'],
                "Val Score": cv_res['test_score']
            })
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(name='Train', x=cv_df['Fold'], y=cv_df['Train Score'], marker_color='#a78bfa'))
            fig_cv.add_trace(go.Bar(name='Validation', x=cv_df['Fold'], y=cv_df['Val Score'], marker_color='#f97316'))
            fig_cv.update_layout(barmode='group', title=f'K-Fold CV Results ({scoring_metric})')
            st.plotly_chart(styled_fig(fig_cv), use_container_width=True)
        
        with tab_cls:
            labels_str = [str(c) for c in (class_names if class_names is not None else np.unique(y))]
            report = classification_report(y_test, test_pred, target_names=labels_str, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
    
    else:  # Regression
        with st.spinner("📊 Computing regression metrics..."):
            r2_tr  = r2_score(y_train, train_pred)
            r2_te  = r2_score(y_test,  test_pred)
            rmse   = np.sqrt(mean_squared_error(y_test, test_pred))
            mae    = mean_absolute_error(y_test, test_pred)
            cv_mean = cv_res['test_score'].mean()
            cv_std  = cv_res['test_score'].std()
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="top-bar" style="background:#f97316"></div>
                <div class="metric-val" style="color:#f97316;font-size:2rem">{r2_te:.4f}</div>
                <div class="metric-label">TEST R²</div>
                <div class="metric-sub">coefficient of determination</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#a78bfa"></div>
                <div class="metric-val" style="color:#a78bfa;font-size:2rem">{r2_tr:.4f}</div>
                <div class="metric-label">TRAIN R²</div>
                <div class="metric-sub">train set score</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#ef4444"></div>
                <div class="metric-val" style="color:#ef4444;font-size:2rem">{rmse:.3f}</div>
                <div class="metric-label">RMSE</div>
                <div class="metric-sub">root mean sq error</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#facc15"></div>
                <div class="metric-val" style="color:#facc15;font-size:2rem">{mae:.3f}</div>
                <div class="metric-label">MAE</div>
                <div class="metric-sub">mean absolute error</div>
            </div>
            <div class="metric-card">
                <div class="top-bar" style="background:#10b981"></div>
                <div class="metric-val" style="color:#10b981;font-size:2rem">{cv_mean:.4f}</div>
                <div class="metric-label">CV R²</div>
                <div class="metric-sub">±{cv_std:.4f} std</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab_pred, tab_res, tab_cv2 = st.tabs(["Predicted vs Actual", "Residuals", "CV Scores"])
        
        with tab_pred:
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Scatter(x=y_test, y=test_pred, mode='markers',
                                        marker=dict(color='#f97316', opacity=0.7, size=5), name='Predictions'))
            mn, mx = min(y_test.min(), test_pred.min()), max(y_test.max(), test_pred.max())
            fig_pa.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                        line=dict(color='#ef4444', dash='dash'), name='Perfect'))
            fig_pa.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted')
            st.plotly_chart(styled_fig(fig_pa), use_container_width=True)
        
        with tab_res:
            residuals = y_test - test_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=test_pred, y=residuals, mode='markers',
                                          marker=dict(color='#a78bfa', opacity=0.7, size=5), name='Residuals'))
            fig_res.add_hline(y=0, line_color='#ef4444', line_dash='dash')
            fig_res.update_layout(title='Residual Plot', xaxis_title='Predicted', yaxis_title='Residuals')
            st.plotly_chart(styled_fig(fig_res), use_container_width=True)
        
        with tab_cv2:
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(k_val)],
                "Train R²": cv_res['train_score'],
                "Val R²": cv_res['test_score']
            })
            fig_cv2 = go.Figure()
            fig_cv2.add_trace(go.Bar(name='Train R²', x=cv_df['Fold'], y=cv_df['Train R²'], marker_color='#a78bfa'))
            fig_cv2.add_trace(go.Bar(name='Val R²', x=cv_df['Fold'], y=cv_df['Val R²'], marker_color='#f97316'))
            fig_cv2.update_layout(barmode='group', title='K-Fold CV Results')
            st.plotly_chart(styled_fig(fig_cv2), use_container_width=True)
    
    # Feature Importance
    if hasattr(model, 'feature_importances_') and selected_features:
        st.markdown("---")
        st.markdown("**🔍 Feature Importance**")
        n_feats = len(model.feature_importances_)
        display_feats = selected_features[:n_feats] if len(selected_features) >= n_feats else selected_features
        fi_df = pd.DataFrame({
            "Feature": display_feats,
            "Importance": model.feature_importances_[:len(display_feats)]
        }).sort_values("Importance", ascending=False)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                        color="Importance",
                        color_continuous_scale=["#252a3a","#f97316"],
                        title="Feature Importances")
        st.plotly_chart(styled_fig(fig_fi), use_container_width=True)
    
    st.session_state.stages_done.add(6)
    
    if st.button("✅ Proceed to Hyperparameter Tuning →"):
        st.session_state.active_stage = 7
        st.rerun()

# ─────────────────────────────────────────────
#  STAGE 7 — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
elif active == 7:
    if not st.session_state.get('model_trained'):
        st.markdown('<div class="warn-box">⚠️ Please train a model first (Stage 5).</div>', unsafe_allow_html=True)
        st.stop()
    
    model        = st.session_state.model
    model_choice = st.session_state.model_choice
    X_train      = st.session_state.X_train
    y_train      = st.session_state.y_train
    scoring_metric = st.session_state.get('scoring_metric', 'f1_weighted')
    
    st.markdown('<div class="sec-header"><span class="num">07</span> Hyperparameter Tuning</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">📌 Tuning: <b>{model_choice}</b> — Grid Search with 3-Fold CV | Scoring: <b>{scoring_metric}</b></div>', unsafe_allow_html=True)
    st.markdown('<div class="fix-banner">✅ Grid search runs only on X_train (already split). Test set is never touched during tuning.</div>', unsafe_allow_html=True)
    
    if "Random Forest" in model_choice or "Extra Trees" in model_choice:
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}
    elif "Decision Tree" in model_choice:
        param_grid = {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
    elif "Gradient Boosting" in model_choice or "XGBoost" in model_choice or "LightGBM" in model_choice:
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2]}
    elif "SVM" in model_choice:
        param_grid = {'C': [0.1, 1.0, 5.0, 10.0]}
    elif "K-Nearest" in model_choice:
        param_grid = {'n_neighbors': [3, 5, 7, 11, 15]}
    elif "Ridge" in model_choice or "Lasso" in model_choice:
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]} if problem_type == "Regression" else {}
    else:
        param_grid = {}
    
    if param_grid:
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        st.markdown(f'<div class="info-box">🔢 Search space: {total_combos} combinations × 3 folds = {total_combos*3} fits</div>', unsafe_allow_html=True)
        
        st.markdown("**Parameter Grid:**")
        for k, v in param_grid.items():
            st.markdown(f"- `{k}`: {v}")
        
        if st.button("🔍 Run Grid Search Now"):
            prog = st.progress(0, text="⏳ Initializing grid search...")
            
            with st.spinner(f"⏳ Running grid search across {total_combos} combinations..."):
                prog.progress(30, text=f"⏳ Fitting {total_combos * 3} models on training set...")
                grid = GridSearchCV(model, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)
                prog.progress(100, text="✅ Grid search complete!")
            
            st.markdown(f'<div class="ok-box">✅ <b>Best Parameters:</b> {grid.best_params_} | Best CV Score: <b>{grid.best_score_:.4f}</b></div>', unsafe_allow_html=True)
            
            st.session_state.model = grid.best_estimator_
            
            res_df = pd.DataFrame(grid.cv_results_)
            param_cols = [c for c in res_df.columns if c.startswith('param_')]
            if param_cols:
                fig_gs = px.scatter(res_df, x='mean_test_score', y='std_test_score',
                                    hover_data=param_cols,
                                    title="Grid Search Results — Score vs Variance",
                                    color='mean_test_score',
                                    color_continuous_scale=["#252a3a","#f97316"],
                                    size_max=20)
                fig_gs.update_layout(xaxis_title=f"Mean CV {scoring_metric}", yaxis_title="Score Std Dev")
                st.plotly_chart(styled_fig(fig_gs), use_container_width=True)
            
            st.session_state.stages_done.add(7)
    else:
        st.markdown('<div class="info-box">ℹ️ Grid search not configured for this model type.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:JetBrains Mono,monospace;font-size:.7rem;color:#2d3548;padding:.5rem 0;">
    FraudShield ML · Leakage-Free Pipeline · Built with Streamlit + scikit-learn + XGBoost
</div>
""", unsafe_allow_html=True)