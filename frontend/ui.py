import streamlit as st
import pandas as pd
from datetime import date
import requests
import os

# Configuration
st.set_page_config(
    page_title="Clinical Lab Analyzer",
    page_icon="🧪",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000/api/predictions/analyze")

# Initialize session state
if 'report_uploaded' not in st.session_state:
    st.session_state.report_uploaded = False
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Sidebar
with st.sidebar:
    st.title("🏥 Clinical Lab")
    st.subheader("AI-Powered Insights")
    
    menu = ["Login", "Analyze", "History"]
    selection = st.sidebar.radio("Navigate", menu)
    
    st.divider()
    st.info("How it works:\n1. Login\n2. Enter Lab Values\n3. Get AI Insights")

# LOGIN PAGE
if selection == "Login":
    st.header("Login")
    
    name = st.text_input("Name")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    dob = st.date_input("Date of Birth", min_value=date(1980, 1, 1))
    
    if st.button("Login"):
        if password == confirm_password and password:
            st.session_state.user_logged_in = True
            st.session_state.user_name = name
            st.success(f"✅ {name} logged in successfully!")
        else:
            st.error("❌ Passwords don't match!")

# ANALYZE PAGE
elif selection == "Analyze":
    if not st.session_state.user_logged_in:
        st.error("❌ Please login first!")
    else:
        st.header("Lab Values Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=150, value=55)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=500, value=150)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=500, value=220)
            hdl = st.number_input("HDL (mg/dL)", min_value=0, max_value=200, value=35)
        
        with col2:
            ldl = st.number_input("LDL (mg/dL)", min_value=0, max_value=500, value=160)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=500, value=180)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.2)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.1)
        
        with col3:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0)
            systolic_bp = st.number_input("Systolic BP", min_value=60, max_value=250, value=135)
            diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=150, value=85)
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing with AI..."):
                lab_values = {
                    "age": float(age),
                    "glucose": float(glucose),
                    "cholesterol": float(cholesterol),
                    "hdl": float(hdl),
                    "ldl": float(ldl),
                    "triglycerides": float(triglycerides),
                    "hemoglobin": float(hemoglobin),
                    "creatinine": float(creatinine),
                    "bmi": float(bmi),
                    "systolic_bp": float(systolic_bp),
                    "diastolic_bp": float(diastolic_bp)
                }
                
                try:
                    response = requests.post(
                        API_URL,
                        json=lab_values,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.analysis_result = result
                        st.success("✅ Analysis Complete!")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure backend is running!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display results
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            st.markdown("---")
            st.subheader("📊 Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            agent2 = result.get('agent2_disease_detection', {})
            col1.metric("🔍 Disease", agent2.get('disease_type', 'N/A'), 
                       f"{agent2.get('confidence', 0)*100:.1f}%")
            
            agent4 = result.get('agent4_risk_assessment', {})
            col2.metric("⚠️ Risk Level", agent4.get('risk_level', 'N/A'),
                       f"{agent4.get('risk_score', 0)}/100")
            
            col3.metric("💊 Confidence", f"{agent4.get('confidence', 0)*100:.1f}%")
            col4.metric("⏱️ Time", f"{result.get('processing_time_ms', 0):.0f}ms")
            
            # Recommendations
            if 'recommendations' in agent4:
                st.subheader("📋 Recommendations")
                for rec in agent4['recommendations']:
                    st.info(rec)
            
            # Trends
            agent3 = result.get('agent3_trend_forecasting', {})

            if agent3:
                st.subheader("📈 Trends Forecast (30 / 60 / 90 Days)")

                for param, forecast in agent3.items():
                    if 'error' not in forecast:
                        current = forecast.get('current', 0)
                        d30 = forecast.get('forecast_30d', 0)
                        d60 = forecast.get('forecast_60d', 0)
                        d90 = forecast.get('forecast_90d', 0)

                        st.markdown(f"### {param.upper()}")

                        # Create dataframe for chart
                        chart_data = pd.DataFrame({
                            "Days": ["Current", "30 Days", "60 Days", "90 Days"],
                            "Values": [current, d30, d60, d90]
                        })

                        st.line_chart(
                            chart_data.set_index("Days")
                        )

                        st.write(
                            f"Trend: **{forecast.get('trend', 'N/A')}**"
                        )

                        st.write("---")

            if st.button("New Analysis"):
                st.session_state.analysis_result = None
                st.rerun()

# HISTORY PAGE
elif selection == "History":
    st.header("Analysis History")
    if not st.session_state.user_logged_in:
        st.error("❌ Please login first!")
    else:
        st.info("📝 History feature coming soon...")