import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

# --- Page Config & Styling ---
st.set_page_config(page_title="AutoML Pipeline Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    .step-header { color: #1E3A8A; font-weight: bold; border-bottom: 2px solid #1E3A8A; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 End-to-End ML Pipeline Dashboard")
st.write("A sophisticated horizontal-flow machine learning environment.")

# --- 1. Problem Type Selection ---
st.sidebar.header("Step 1: Configuration")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])

# --- 2. Data Input & Visualization ---
st.header("📂 Data Acquisition & PCA Analysis")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### Data Preview")
        st.dataframe(df.head(10))
        target_col = st.selectbox("Select Target Feature", df.columns)
        features = st.multiselect("Select Features for Analysis", [c for c in df.columns if c != target_col], default=[c for c in df.columns if c != target_col])

    with col2:
        if len(features) >= 2:
            st.write("### Interactive Data Shape (PCA)")
            temp_df = df[features].dropna().select_dtypes(include=[np.number])
            pca = PCA(n_components=2)
            components = pca.fit_transform(StandardScaler().fit_transform(temp_df))
            fig = px.scatter(components, x=0, y=1, color=df.loc[temp_df.index, target_col], 
                             labels={'0': 'PC1', '1': 'PC2'}, title="2D PCA Projection")
            st.plotly_chart(fig, use_container_width=True)

    # --- 3. Exploratory Data Analysis (EDA) ---
    st.markdown("---")
    st.header("📊 Exploratory Data Analysis")
    with st.expander("Show EDA Results"):
        c1, c2 = st.columns(2)
        c1.write("**Descriptive Statistics**")
        c1.dataframe(df.describe())
        c2.write("**Missing Values**")
        c2.write(df.isnull().sum())
        
        corr = df.select_dtypes(include=[np.number]).corr()
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"))

    # --- 4. Data Engineering & Cleaning ---
    st.markdown("---")
    st.header("🛠️ Data Engineering & Outlier Removal")
    
    # Imputation
    impute_method = st.selectbox("Imputation Method", ["mean", "median", "most_frequent"])
    if st.button("Apply Imputation"):
        num_cols = df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy=impute_method)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        st.success("Missing values handled.")

    # Outlier Detection
    outlier_method = st.selectbox("Outlier Detection Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
    outlier_idx = []

    if outlier_method == "IQR":
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1
        outlier_idx = df[((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)].index
    elif outlier_method == "Isolation Forest":
        iso = IsolationForest(contamination=0.05)
        preds = iso.fit_predict(df[features].select_dtypes(include=[np.number]))
        outlier_idx = df.index[preds == -1]

    st.warning(f"Detected {len(outlier_idx)} outliers using {outlier_method}.")
    if len(outlier_idx) > 0:
        if st.button("Delete Selected Outliers"):
            df = df.drop(outlier_idx)
            st.success("Outliers removed from the session!")

    # --- 5. Feature Selection ---
    st.markdown("---")
    st.header("🎯 Feature Selection")
    fs_method = st.multiselect("Select Methods", ["Variance Threshold", "Information Gain"])
    
    selected_features = features
    if "Variance Threshold" in fs_method:
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(df[features].select_dtypes(include=[np.number]))
        selected_features = df[features].columns[selector.get_support()].tolist()
        st.write(f"Features after Variance Threshold: {selected_features}")

    # --- 6. Data Split & Model Selection ---
    st.markdown("---")
    st.header("🤖 Model Training")
    
    test_size = st.slider("Test Set Size (%)", 10, 50, 20)
    model_choice = st.selectbox("Select Model", ["Linear/Logistic Regression", "SVM", "Random Forest"])
    
    k_val = st.number_input("K-Fold Cross Validation (K)", min_value=2, max_value=10, value=5)

    # Prepare Data
    X = df[selected_features]
    y = df[target_col]
    if problem_type == "Classification":
        y = LabelEncoder().fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    # Initialize Model
    model = None
    if model_choice == "Linear/Logistic Regression":
        model = LogisticRegression() if problem_type == "Classification" else LinearRegression()
    elif model_choice == "SVM":
        kernel = st.radio("Kernel", ["linear", "poly", "rbf"])
        model = SVC(kernel=kernel) if problem_type == "Classification" else SVR(kernel=kernel)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier() if problem_type == "Classification" else RandomForestRegressor()

    if st.button("Train and Validate"):
        # K-Fold
        cv_results = cross_validate(model, X_train, y_train, cv=k_val, return_train_score=True)
        
        # Final Fit
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # --- 7. Performance Metrics ---
        st.subheader("📈 Performance Metrics")
        m1, m2, m3 = st.columns(3)
        
        if problem_type == "Classification":
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            m1.metric("Train Accuracy", f"{train_acc:.2%}")
            m2.metric("Test Accuracy", f"{test_acc:.2%}")
            # Overfit check
            if train_acc - test_acc > 0.15:
                st.error("Model is likely Overfitting!")
            elif train_acc < 0.6:
                st.warning("Model is likely Underfitting!")
        else:
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            m1.metric("Train R2", f"{train_r2:.2f}")
            m2.metric("Test R2", f"{test_r2:.2f}")

        st.write("**K-Fold Validation Scores:**", cv_results['test_score'])

    # --- 8. Hyperparameter Tuning ---
    st.markdown("---")
    st.header("⚙️ Hyperparameter Tuning")
    if st.checkbox("Perform Grid Search"):
        if model_choice == "Random Forest":
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
            grid = GridSearchCV(model, param_grid, cv=3)
            grid.fit(X_train, y_train)
            st.write("Best Parameters:", grid.best_params_)
            st.write("Best Score Improvement:", grid.best_score_)
        else:
            st.info("Tuning logic currently optimized for Random Forest in this demo.")

else:
    st.info("Please upload a CSV file to begin.")