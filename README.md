🏡 California Housing Price Prediction Dashboard
A Machine Learning + Explainable AI Project (Streamlit Web App)

This project is a fully interactive House Price Prediction Dashboard built using Streamlit, featuring:

🌲 Random Forest, 🌄 Gradient Boosting, and 🚀 XGBoost models

🔧 A complete Scikit-Learn preprocessing pipeline

🧠 SHAP explainability (Waterfall, Force Plot, Global Summary)

🎨 Professional Dark Mode UI (VS Code / GitHub theme)

📊 Interactive charts, feature importance, and data insights

🧩 Fully deployable on Streamlit Cloud

📌 Features
🏠 1. Predict House Prices

Enter the following features:

Longitude / Latitude

Housing Median Age

Median Income

Rooms per Household

Population per Household

Ocean Proximity (One-Hot Encoded)

The app handles preprocessing internally using the saved ML pipeline.

📈 2. Global Model Insights

Feature importance (from tree-based models)

Distribution plots

Income vs House Price visualization

Top features graphically ranked

🧠 3. Explainable AI (SHAP)

Includes:

SHAP Waterfall Plot

SHAP Force Plot (HTML embedded)

Global SHAP Mean |Value| Summary

This allows deep inspection of why a model predicted a price.

⚙️ 4. Model Details

Pipeline steps (preprocessing + regressor)

Extracted feature names

Input schema

Model metadata (Random Forest / Gradient Boosting / XGBoost)

🗂 Project Structure
house-price-dashboard/
│
├── app.py                         # Main Streamlit App
├── housing.csv                    # Dataset
├── final_model.pkl                # Random Forest Model
├── gradient_boosting_model.pkl    # Gradient Boosting Model
├── xgboost_model.pkl              # XGBoost Model
├── requirements.txt               # Dependencies for Streamlit Cloud
└── README.md                      # Project Documentation

🧪 Models & Training Overview

Models were trained using a Scikit-Learn Pipeline, ensuring:

consistent preprocessing

correct feature ordering

reproducibility

smooth deployment

Feature transformations include:

StandardScaler

OneHotEncoder

Feature engineering:

rooms_per_household

population_per_household