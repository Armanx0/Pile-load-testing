# 🏗️ Pile Bearing Capacity AI Suite

A professional, machine learning-powered web application designed for geotechnical engineers to predict the ultimate bearing capacity of piles. This tool utilizes four distinct **XGBoost** regression models trained on various soil and pile datasets. 

Crucially, it bridges the gap between modern AI and traditional geotechnical engineering by directly comparing the Machine Learning predictions against standard **IS Code empirical formulas** in real-time.

---

## 🌟 Key Features
- **4 Specialized ML Models:** Tailored for different soil conditions (Layered SPT, Cohesive-Friction, Cohesionless Effective Stress, Undrained Shear).
- **Live IS Code Benchmarking:** Instantly compares the AI's prediction against conservative Meyerhof-type and IS code calculations.
- **Dynamic Visualizations:** Generates side-by-side bar charts comparing Empirical Capacity, ML Capacity, and Actual Field Test results (when provided).
- **Hyperparameter-Optimized:** All models are fine-tuned using `RandomizedSearchCV` with 10-fold cross-validation.
- **Decoupled Architecture:** Clean, modular codebase separating the Streamlit UI, metric charting, and model training logic.

---

## 📂 Project Structure

```text
pile-capacity-app/
│
├── app.py                      # Main entry point for the Streamlit web application
├── utils.py                    # Shared UI utilities and dynamic charting logic
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (venv, __pycache__, etc.)
├── check_cols.py               # Utility script to debug CSV column encodings
│
├── models/                     # ML Model folders (training scripts & serialized files)
│   ├── model1/
│   │   ├── train_model1.py     # Script to train the SPT Layered model
│   │   ├── scaler.pkl          # RobustScaler for Model 1
│   │   └── xgboost_model.pkl   # Trained XGBoost model 1
│   ├── model2/
│   │   ├── train_model2.py
│   │   ├── scaler.pkl
│   │   └── xgboost_model.pkl
│   ├── model3/
│   │   ├── train_model3.py
│   │   ├── scaler.pkl
│   │   └── xgboost_model.pkl
│   └── model4/
│       ├── train_model4.py
│       ├── scaler.pkl
│       └── xgboost_model.pkl
│
└── tabs/                       # Decoupled Streamlit UI logic
    ├── __init__.py
    ├── tab1_model1.py          # UI and IS Code math for Model 1
    ├── tab2_model2.py          # UI and IS Code math for Model 2
    ├── tab3_model3.py          # UI and IS Code math for Model 3
    └── tab4_model4.py          # UI and IS Code math for Model 4
