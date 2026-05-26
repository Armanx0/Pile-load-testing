import streamlit as st
import pandas as pd
import joblib
import math
import shap
import matplotlib.pyplot as plt
from tabs.tab1_model1 import display_results

@st.cache_resource
def load_assets():
    scaler = joblib.load('models/model3/scaler.pkl')
    model = joblib.load('models/model3/xgboost_model.pkl')
    return scaler, model

def render():
    st.header("Cohesionless Effective Stress Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        pile_area = st.number_input("Cross section area m2", value=0.28274, step=0.01, key="m3_area")
        friction_angle = st.number_input("Friction angle ϕ (deg)", value=36.0, step=0.5, key="m3_phi")
    with col2:
        pile_length = st.number_input("Embedded Depth (m)", value=17.5, step=0.5, key="m3_l")
        effective_stress = st.number_input("Effective Stress σv (kPa)", value=127.0, step=1.0, key="m3_sig")

    actual_test_value = st.number_input("Actual Qmax (kN) [Optional]", min_value=0.0, value=2128.69, step=50.0, key="m3_act")

    if st.button("Predict Model 3 Capacity", type="primary", key="btn3"):
        scaler, model = load_assets()
        
        input_data = pd.DataFrame([{
            'Cross section area m2': pile_area,
            'Embedded Depth (m)': pile_length,
            'Friction angle ϕ (deg)': friction_angle,
            "Effective Stress σv' (kPa)": effective_stress
        }])
        ml_pred = model.predict(scaler.transform(input_data))[0]
        
        # IS Code Calculation
        phi_rad = math.radians(friction_angle)
        
        D = math.sqrt((4 * pile_area) / math.pi)
        A_s = math.pi * D * pile_length
        K = 1 - math.sin(phi_rad)
        
        N_q = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45) + (phi_rad / 2)) ** 2)
        delta_rad = math.radians(0.67 * friction_angle)
        
        Q_p = pile_area * effective_stress * N_q
        Q_s = A_s * K * effective_stress * math.tan(delta_rad)
        
        is_pred = Q_p + Q_s
        
        display_results(actual_test_value, ml_pred, is_pred, model, scaler, input_data)