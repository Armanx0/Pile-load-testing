import streamlit as st
import pandas as pd
import joblib
import math
import shap
import matplotlib.pyplot as plt
import numpy as np

if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.bool = bool

@st.cache_resource
def load_assets():
    scaler = joblib.load('models/model1/scaler.pkl')
    model = joblib.load('models/model1/xgboost_model.pkl')
    return scaler, model

def render():
    st.header("SPT Layered Soil Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        diameter = st.number_input("Diameter D (mm)", value=400.0, step=50.0, key="m1_d")
        ground_elev = st.number_input("Ground elevation Xg (m)", value=10.0, step=0.5, key="m1_xg")
        pile_top_elev = st.number_input("Pile top elevation Xp (m)", value=10.0, step=0.5, key="m1_xp")
        spt_shaft = st.number_input("SPT blow count at shaft Ns", value=11.75, step=0.25, key="m1_ns")
        depth_x1 = st.number_input("Depth of 1st layer X1 (m)", value=3.45, step=0.5, key="m1_x1")
        depth_x3 = st.number_input("Depth of 3rd layer X3 (m)", value=0.3, step=0.5, key="m1_x3")
    with col2:
        pile_tip_elev = st.number_input("Pile tip elevation Xm (m)", value=-1.75, step=0.5, key="m1_xm")
        extra_top_elev = st.number_input("Extra pile top elevation Xt (m)", value=0.0, step=0.5, key="m1_xt")
        spt_tip = st.number_input("SPT blow count at tip Nt", value=7.59, step=0.25, key="m1_nt")
        depth_x2 = st.number_input("Depth of 2nd layer X2 (m)", value=8.0, step=0.5, key="m1_x2")

    actual_test_value = st.number_input("Actual Static Load Test (kN) [Optional]", min_value=0.0, value=0.0, step=50.0, key="m1_act")

    if st.button("Predict Model 1 Capacity", type="primary", key="btn1"):
        try:
            scaler, model = load_assets()
            
            # ML Prediction
            input_data = pd.DataFrame([{
                'Diameter D (mm)': diameter,
                'Depth of first layer of soil embedded X1 (m)': depth_x1,
                'Depth of second layer of soil embedded X2 (m)': depth_x2,
                'Depth of third layer of soil embedded X3 (m)': depth_x3,
                'Pile top elevation Xp (m)': pile_top_elev,
                'Ground elevation Xg (m)': ground_elev,
                'Extra pile top elevation Xt (m)': extra_top_elev,
                'Pile tip elevation Xm (m)': pile_tip_elev,
                'SPT blow count at pile shaft Ns': spt_shaft,
                'SPT blow count at pile tip Nt': spt_tip
            }])
            ml_pred = model.predict(scaler.transform(input_data))[0]
            
            # IS Code Calculation
            D_m = diameter / 1000.0
            L = depth_x1 + depth_x2 + depth_x3
            A_p = (math.pi * (D_m ** 2)) / 4.0
            P = math.pi * D_m
            Q_b = A_p * (400.0 * spt_tip)
            Q_s = P * L * (2.0 * spt_shaft)
            is_pred = Q_b + Q_s
            
            display_results(actual_test_value, ml_pred, is_pred, model, scaler, input_data)
        except Exception as e:
            st.error(f"Error loading models or calculating. Ensure model1 .pkl files exist. ({e})")

# Shared function for UI display (keeps code clean)
def display_results(actual, ml_pred, is_pred, model, scaler, input_data):
    st.markdown("---")
    
    # 1. Display Metrics and Define Chart Data
    if actual > 0:
        ml_diff = ((ml_pred - actual) / actual) * 100
        is_diff = ((is_pred - actual) / actual) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("🏗️ Actual Test", f"{actual:,.2f} kN")
        c2.metric("🤖 ML Capacity", f"{ml_pred:,.2f} kN", f"{ml_diff:.1f}% vs Actual", delta_color="inverse")
        c3.metric("📐 IS Code Capacity", f"{is_pred:,.2f} kN", f"{is_diff:.1f}% vs Actual", delta_color="inverse")
        
        # Define variables for the chart (3 bars)
        labels = ['IS Code (Empirical)', 'ML Predicted', 'Actual Field Test']
        values = [is_pred, ml_pred, actual]
        colors = ['#e74c3c', '#3498db', '#2ecc71'] # Red, Blue, Green
        
    else:
        is_diff = ((is_pred - ml_pred) / ml_pred) * 100
        c1, c2 = st.columns(2)
        c1.metric("🤖 ML Capacity", f"{ml_pred:,.2f} kN")
        c2.metric("📐 IS Code Capacity", f"{is_pred:,.2f} kN", f"{is_diff:.1f}% vs ML", delta_color="normal")
        
        # Define variables for the chart (2 bars)
        labels = ['IS Code (Empirical)', 'ML Predicted']
        values = [is_pred, ml_pred]
        colors = ['#e74c3c', '#3498db'] # Red, Blue

    # 2. Display Comparison Bar Chart
    st.markdown("### 📊 Capacity Comparison")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create horizontal bars using the variables defined above
    bars = ax.barh(labels, values, color=colors, height=0.6)
    
    # Add the exact value text to the end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max(values) * 0.02), # Slightly offset from the end of the bar
            bar.get_y() + bar.get_height() / 2,
            f'{width:,.1f} kN',
            ha='left',
            va='center',
            fontweight='bold',
            fontsize=11
        )
    
    # Clean up the chart appearance
    ax.set_xlabel('Pile Bearing Capacity (kN)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=12) # Hide y-axis ticks but keep labels
    
    # Add a buffer to the x-axis so the text labels don't get cut off
    ax.set_xlim(0, max(values) * 1.2)
    
    st.pyplot(fig, use_container_width=True)