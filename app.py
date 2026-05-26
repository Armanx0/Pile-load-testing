import streamlit as st
from tabs import tab1_model1, tab2_model2, tab3_model3, tab4_model4

# Page configuration
st.set_page_config(page_title="Pile Capacity AI Suite", page_icon="🏗️", layout="wide")

st.title("🏗️ Pile Bearing Capacity AI Suite")
st.write("Compare 4 specialized Machine Learning models against IS Code empirical formulas.")

# Create the 4 tabs
t1, t2, t3, t4 = st.tabs([
    "Model 1: SPT Layered Method", 
    "Model 2: Cohesive-Friction Method", 
    "Model 3: Cohesionless Effective Stress", 
    "Model 4: Undrained Shear Method"
])

# Render each tab from its decoupled file
with t1:
    tab1_model1.render()
with t2:
    tab2_model2.render()
with t3:
    tab3_model3.render()
with t4:
    tab4_model4.render()