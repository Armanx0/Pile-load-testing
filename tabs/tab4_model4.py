import streamlit as st
import pandas as pd
import joblib
import math
import shap
import matplotlib.pyplot as plt
from tabs.tab1_model1 import display_results

@st.cache_resource
def load_assets():
    scaler = joblib.load('models/model4/scaler.pkl')
    model = joblib.load('models/model4/xgboost_model.pkl')
    return scaler, model

def render():
    st.header("Undrained Shear Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        pile_area = st.number_input("Cross section area m2", value=0.0254469, step=0.001, key="m4_area")
        cu = st.number_input("Undrained Shear Strength Cu (kPa)", value=47.0, step=1.0, key="m4_cu")
    with col2:
        pile_length = st.number_input("Total Pile Length L (m)", value=1.8, step=0.1, key="m4_l")
        
    actual_test_value = st.number_input("Actual Qmax (kN) [Optional]", min_value=0.0, value=0.0, step=10.0, key="m4_act")

    if st.button("Predict Model 4 Capacity", type="primary", key="btn4"):
        scaler, model = load_assets()
        
        input_data = pd.DataFrame([{
            'Cross section area m2': pile_area,
            'Total Pile Length L (m)': pile_length,
            'Undrained Shear Strength Cu (kPa)': cu
        }])
        ml_pred = model.predict(scaler.transform(input_data))[0]
        
        # IS Code Calculation
        alpha = 0.5 * math.sqrt(100.0 / cu)
        D = math.sqrt((4 * pile_area) / math.pi)
        A_s = math.pi * D * pile_length
        N_c = 9.0
        
        Q_p = pile_area * N_c * cu
        Q_s = alpha * cu * A_s
        
        is_pred = Q_p + Q_s
        
        display_results(actual_test_value, ml_pred, is_pred, model, scaler, input_data)