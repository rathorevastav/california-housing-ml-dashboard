# final_professional_app.py (paste over app.py)
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import shap
import matplotlib.pyplot as plt


st.write('Files in directory:', os.listdir())
st.write('Absolute working directory:', os.getcwd())
# ---------------------------
# Page + CSS (light professional)
# ---------------------------
st.set_page_config(page_title="House Price Dashboard", layout="wide", initial_sidebar_state="expanded")
# ---------------------------
# DARK MODE (Professional VS Code Theme)
# ---------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117 !important;
            color: white !important;
        }
        .block-container {
            background-color: #0e1117 !important;
        }
        .sidebar .sidebar-content {
            background-color: #0e1117 !important;
        }
        h1, h2, h3, h4, h5, h6, label, p, span, div {
            color: white !important;
        }
        .stMarkdown, .stTextInput, .css-1cpxqw2, .css-1offfwp {
            color: white !important;
        }
        .stMetric {
            background-color: #1a1f25 !important;
            border-radius: 10px;
            padding: 10px;
        }
        .card {
            background: #1a1f25 !important;
            border-radius: 10px;
            padding: 12px 18px;
            box-shadow: 0 1px 3px rgba(255,255,255,0.1);
        }
        input, select, textarea {
            color: white !important;
            background-color: #1a1f25 !important;
        }
        .stDataFrame {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🏡 California Housing Price — Professional Dashboard")
st.markdown("**Portfolio demo:** prediction, model comparison, feature importance and SHAP explainability.")

# ---------------------------------------------------------
# Load background data (cached)
# ---------------------------------------------------------
@st.cache_data
def load_background_df(path="housing.csv"):
    df = pd.read_csv(path)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"] / df["households"]
    df.drop(["total_rooms", "total_bedrooms", "population", "households"], axis=1, inplace=True)
    return df

housing_df = load_background_df("housing.csv")
X = housing_df.drop("median_house_value", axis=1)

# numeric cols must match notebook
numeric_cols = [
    "longitude",
    "latitude",
    "housing_median_age",
    "median_income",
    "rooms_per_household",
    "population_per_household"
]

# ---------------------------------------------------------
# Safe model loader (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

# expected names (edit if you used different names)
RF_PATH = "final_model.pkl"
GB_PATH = "gradient_boosting_model.pkl"
XGB_PATH = "xgboost_model.pkl"

rf_model = load_model(RF_PATH)
gb_model = load_model(GB_PATH)
xgb_model = load_model(XGB_PATH)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Model Controls")
model_choice = st.sidebar.selectbox(
    "Select model",
    ("Random Forest", "Gradient Boosting", "XGBoost")
)

st.sidebar.markdown("**Model files present**")
st.sidebar.write(f"• Random Forest: `{RF_PATH}` — {'✅' if rf_model else '❌ missing'}")
st.sidebar.write(f"• Gradient Boosting: `{GB_PATH}` — {'✅' if gb_model else '❌ missing'}")
st.sidebar.write(f"• XGBoost: `{XGB_PATH}` — {'✅' if xgb_model else '❌ missing'}")

if st.sidebar.button("Reload (rerun)"):
    st.experimental_rerun()

# assign model object
if model_choice == "Random Forest":
    model = rf_model
    model_path = RF_PATH
elif model_choice == "Gradient Boosting":
    model = gb_model
    model_path = GB_PATH
else:
    model = xgb_model
    model_path = XGB_PATH

# ---------------------------------------------------------
# Top metrics (cards)
# ---------------------------------------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.write("<div class='card'>", unsafe_allow_html=True)
    st.metric("Dataset rows", f"{housing_df.shape[0]:,}")
    st.write("</div>", unsafe_allow_html=True)
with c2:
    st.write("<div class='card'>", unsafe_allow_html=True)
    st.metric("Avg house price", f"${housing_df['median_house_value'].mean():,.0f}")
    st.write("</div>", unsafe_allow_html=True)
with c3:
    st.write("<div class='card'>", unsafe_allow_html=True)
    st.metric("Model loaded", "Yes" if model else "No")
    st.write("</div>", unsafe_allow_html=True)
with c4:
    st.write("<div class='card'>", unsafe_allow_html=True)
    st.metric("Selected model", model_choice)
    st.write("</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_dashboard, tab_predict, tab_shap, tab_model = st.tabs(
    ["📊 Dashboard", "🏠 Prediction", "🧠 Explainability", "⚙️ Model Details"]
)

# ------------------ Dashboard tab ------------------
with tab_dashboard:
    st.header("Dashboard Overview")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Median House Value Distribution")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(housing_df["median_house_value"], bins=30)
        ax.set_xlabel("Median House Value")
        st.pyplot(fig)
    with col2:
        st.subheader("Median Income vs Price (sample)")
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        sample = housing_df.sample(500, random_state=1)
        ax2.scatter(sample["median_income"], sample["median_house_value"], alpha=0.6, s=10)
        ax2.set_xlabel("Median Income")
        ax2.set_ylabel("House Value")
        st.pyplot(fig2)

    st.markdown("### Top features (from selected model)")
    if model is None:
        st.info("No model loaded; place the .pkl model files in the project folder and reload.")
    else:
        try:
            final_est = model.named_steps["regressor"]
            pre = model.named_steps["preprocessor"]
            # Attempt to extract feature names
            try:
                feat_names = list(pre.get_feature_names_out())
            except Exception:
                # robust fallback: numeric_cols + ohe features if present
                try:
                    ohe = pre.named_transformers_.get("cat", None)
                    if ohe is not None:
                        ohe_feats = list(ohe.get_feature_names_out(["ocean_proximity"]))
                    else:
                        ohe_feats = []
                    feat_names = numeric_cols + ohe_feats
                except Exception:
                    feat_names = numeric_cols

            if hasattr(final_est, "feature_importances_"):
                fi = final_est.feature_importances_
                fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(10)
                st.bar_chart(fi_df.set_index("feature"))
            else:
                st.info("Selected model does not expose feature_importances_.")
        except Exception as e:
            st.error(f"Top features error: {e}")

# ------------------ Prediction tab ------------------
with tab_predict:
    st.header("Make a Prediction")
    st.markdown("Enter raw feature values — the pipeline handles preprocessing (scaling / encoding).")

    with st.form("predict_form"):
        a, b = st.columns(2)
        with a:
            longitude = st.number_input("Longitude", value=-122.0, format="%.5f")
            latitude = st.number_input("Latitude", value=37.0, format="%.5f")
            housing_median_age = st.number_input("Housing Median Age", value=20.0, format="%.1f")
            median_income = st.number_input("Median Income", value=4.0, format="%.3f")
        with b:
            rooms_per_household = st.number_input("Rooms per Household", value=5.0, format="%.2f")
            population_per_household = st.number_input("Population per Household", value=3.0, format="%.2f")
            ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

        submit = st.form_submit_button("Predict")

    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,
        "population_per_household": population_per_household,
        "ocean_proximity": ocean_proximity
    }])

    st.subheader("Input preview")
    st.dataframe(input_df)

    if submit:
        if model is None:
            st.error("No model loaded. Place model .pkl in folder and press Reload.")
        else:
            try:
                pred = model.predict(input_df)[0]
                st.success(f"🏠 Predicted house price (using **{model_choice}**) : **${pred:,.2f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------ Explainability tab ------------------
with tab_shap:
    st.header("Explainability & SHAP")
    st.write("SHAP computations are heavy — we use a small background sample and compute SHAP only when requested to keep the app snappy.")

    if model is None:
        st.info("Load a model first.")
    else:
        try:
            pre = model.named_steps["preprocessor"]
            reg = model.named_steps["regressor"]

            # BACKGROUND sample size reduced for speed
            bg_sample_n = 50
            X_bg_raw = X.sample(bg_sample_n, random_state=42)
            X_bg = pre.transform(X_bg_raw)

            # Choose which input to explain: last input in prediction tab or random sample
            if 'input_df' in locals() and input_df is not None:
                raw_input = input_df
            else:
                raw_input = X_bg_raw.iloc[[0]]

            X_input = pre.transform(raw_input)

            st.subheader("Single prediction SHAP (waterfall + force)")

            # compute explainer and shap on demand (button)
            if st.button("Compute SHAP explanation (may take a few seconds)"):
                with st.spinner("Computing SHAP explanation..."):
                    # Build TreeExplainer and compute shap values
                    explainer = shap.TreeExplainer(reg)
                    shap_values = explainer.shap_values(X_input)

                    # expected value normalization
                    expected_val = explainer.expected_value
                    if isinstance(expected_val, (list, tuple, np.ndarray)):
                        expected_val = expected_val[0]
                    shap_vector = shap_values[0] if isinstance(shap_values, list) else shap_values

                    # Waterfall (legacy)
                    st.write("#### SHAP Waterfall (single prediction)")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots._waterfall.waterfall_legacy(expected_val, shap_vector[0], feature_names=list(pre.get_feature_names_out()))
                    st.pyplot(fig)

                    # Force plot (legacy HTML)
                    st.write("#### SHAP Force Plot (single prediction)")
                    try:
                        force_html = shap.force_plot(expected_val, shap_vector[0], feature_names=list(pre.get_feature_names_out()), matplotlib=False)
                        shap_html_path = "force_plot.html"
                        shap.save_html(shap_html_path, force_html)
                        with open(shap_html_path, "r", encoding="utf-8") as f:
                            html = f.read()
                        st.components.v1.html(html, height=340, scrolling=True)
                    except Exception as e:
                        st.error(f"Force plot generation failed: {e}")

                    # SHAP summary (mean abs) on background
                    st.write("#### SHAP Summary (mean |SHAP| across background)")
                    try:
                        shap_bg = explainer.shap_values(X_bg)
                        shap_arr = shap_bg[0] if isinstance(shap_bg, list) else shap_bg
                        mean_abs = np.mean(np.abs(shap_arr), axis=0)
                        feat_names = list(pre.get_feature_names_out())
                        sdf = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).head(10)
                        st.bar_chart(sdf.set_index("feature"))
                    except Exception as e:
                        st.info("Global SHAP summary failed.")
            else:
                st.info("Press the button above to compute SHAP explanation for the selected input (runs once per click).")

        except Exception as e:
            st.error(f"SHAP explanation error: {e}")

# ------------------ Model details tab ------------------
with tab_model:
    st.header("Model Details & Metadata")
    st.markdown(f"**Selected model:** {model_choice}")
    st.markdown(f"- Model file: `{model_path}`")
    st.markdown(f"- Model loaded: {'✅' if model else '❌'}")

    if model:
        try:
            st.subheader("Pipeline steps")
            for name, step in model.steps:
                st.write(f"- **{name}**: `{type(step).__name__}`")

            st.subheader("Feature names used by the pipeline")
            try:
                pre = model.named_steps["preprocessor"]
                fnames = list(pre.get_feature_names_out())
                st.write(f"Total features after preprocessing: {len(fnames)}")
                st.dataframe(pd.DataFrame({"feature": fnames}))
            except Exception as e:
                # robust fallback to construct from numeric + OHE features
                try:
                    pre = model.named_steps["preprocessor"]
                    ohe = pre.named_transformers_.get("cat", None)
                    if ohe is not None:
                        ohe_feats = list(ohe.get_feature_names_out(["ocean_proximity"]))
                    else:
                        ohe_feats = []
                    fnames = numeric_cols + ohe_feats
                    st.write(f"Total features after fallback preprocessing: {len(fnames)}")
                    st.dataframe(pd.DataFrame({"feature": fnames}))
                except Exception as e2:
                    st.error("Could not extract feature names from the pipeline.")
        except Exception as e:
            st.error(f"Could not display model details: {e}")
    else:
        st.info("No model loaded — put the model .pkl in the project folder and press Reload (sidebar).")

# Footer: show uploaded notebook path (so you can reference it for your repo)
st.markdown("---")
st.markdown("Built for portfolio / resume • California housing dataset • SHAP explainability included")
st.markdown("Notebook (uploaded): `/mnt/data/vertopal.com_house price predictor project.pdf`")
