import streamlit as st
import pandas as pd
import joblib
import math
import shap
import matplotlib.pyplot as plt
from tabs.tab1_model1 import display_results # Reuse the UI function

@st.cache_resource
def load_assets():
    scaler = joblib.load('models/model2/scaler.pkl')
    model = joblib.load('models/model2/xgboost_model.pkl')
    return scaler, model

def render():
    st.header("Cohesive-Friction Soil Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        cohesion = st.number_input("Average Cohesion (kN/m2)", value=33.0, step=1.0)
        friction_angle = st.number_input("Average Friction angle (°)", value=28.85, step=0.1)
        pile_area = st.number_input("Pile Area (m2)", value=0.1, step=0.01)
    with col2:
        unit_weight = st.number_input("Average soil Specific weight (kN/m3)", value=9.82, step=0.1)
        pile_soil_friction = st.number_input("Average Pile-Soil friction angle (°)", value=12.49, step=0.1)
        pile_length = st.number_input("Pile Length (m)", value=19.5, step=0.5)

    actual_test_value = st.number_input("Actual Pile Capacity (kN) [Optional]", min_value=0.0, value=1040.0, step=50.0, key="m2_act")

    if st.button("Predict Model 2 Capacity", type="primary", key="btn2"):
        scaler, model = load_assets()
        
        # 1. ML Prediction
        input_data = pd.DataFrame([{
            'Average Cohesion (kN/m2)': cohesion,
            'Average Friction angle (°)': friction_angle,
            'Average soil Specific weight (kN/m3)': unit_weight,
            'Average Pile-Soil friction angle (°)': pile_soil_friction,
            'Pile Area (m2)': pile_area,
            'Pile Length (m)': pile_length
        }])
        ml_pred = model.predict(scaler.transform(input_data))[0]
        
        # 2. IS Code Calculation (Translating your math exactly)
        phi_rad = math.radians(friction_angle)
        delta_rad = math.radians(pile_soil_friction)
        
        D = math.sqrt((4 * pile_area) / math.pi)
        A_s = math.pi * D * pile_length
        
        N_q = math.exp(math.pi * math.tan(phi_rad)) * (math.tan(math.radians(45) + (phi_rad / 2)) ** 2)
        N_gamma = 2 * (N_q + 1) * math.tan(phi_rad)
        N_c = 9.0
        
        D_c = 15 * D
        sigma_v = (unit_weight * D_c) / 2.0
        P_0 = unit_weight * D_c
        
        q_p = (cohesion * N_c) + (P_0 * N_q) + (0.5 * unit_weight * D * N_gamma)
        Q_p = q_p * pile_area
        
        Q_sc = 1.0 * cohesion * A_s
        K = 1 - math.sin(phi_rad)
        Q_sf = K * sigma_v * math.tan(delta_rad) * A_s
        
        is_pred = Q_p + Q_sc + Q_sf
        
        display_results(actual_test_value, ml_pred, is_pred, model, scaler, input_data)