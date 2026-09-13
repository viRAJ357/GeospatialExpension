import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Geospatial Crop Monitoring", layout="wide")

st.title("🌱 Geospatial Crop Monitoring Dashboard")
st.markdown("Dynamic calculations based on real Kaggle agricultural dataset.")

@st.cache_resource
def load_models():
    try:
        models = {
            'yield': joblib.load('crop_yield_prediction_model.pkl'),
            'disease': joblib.load('disease_detection_model.pkl'),
            'irrigation': joblib.load('irrigation_recommendation_model.pkl'),
            'nitrogen': joblib.load('nitrogen_recommendation_model.pkl'),
            'soil': joblib.load('soil_health_model.pkl'),
            'water': joblib.load('water_stress_model.pkl')
        }
        features = joblib.load('feature_columns.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return models, features, scaler
    except Exception as e:
        st.error(f"Models are still training or not found: {e}")
        return None, None, None

models, feature_cols, scaler = load_models()

st.sidebar.header("Input Parameters")
if feature_cols:
    input_data = {}
    for col in feature_cols:
        if 'celsius' in col.lower() or 'temp' in col.lower():
            input_data[col] = st.sidebar.slider(col, 0.0, 50.0, 25.0)
        elif 'percent' in col.lower() or 'humidity' in col.lower():
            input_data[col] = st.sidebar.slider(col, 0.0, 100.0, 50.0)
        elif 'ph' in col.lower():
            input_data[col] = st.sidebar.slider(col, 0.0, 14.0, 7.0)
        elif 'kg_ha' in col.lower() or 'ppm' in col.lower():
            input_data[col] = st.sidebar.number_input(col, min_value=0.0, value=50.0)
        else:
            input_data[col] = st.sidebar.number_input(col, value=1.0)
    
    if st.button("Calculate Predictions"):
        df_input = pd.DataFrame([input_data])
        # Depending on how the original notebook scaled things:
        # We try to apply the scaler if it exists and handles these columns
        try:
            # Assuming the scaler expects the exact feature_cols
            X_scaled = scaler.transform(df_input)
        except Exception:
            # Fallback if scaler logic is slightly different
            X_scaled = df_input
            
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🌾 Crop Yield")
            pred_yield = models['yield'].predict(X_scaled)[0]
            st.metric("Predicted Yield (tons/ha)", f"{pred_yield:.2f}")
            
            st.subheader("🦠 Disease Detection")
            pred_dis = models['disease'].predict(X_scaled)[0]
            st.metric("Disease Risk / Status", str(pred_dis))
            
        with col2:
            st.subheader("💧 Irrigation")
            pred_irr = models['irrigation'].predict(X_scaled)[0]
            st.metric("Recommendation", str(pred_irr))
            
            st.subheader("🧪 Nitrogen")
            pred_nit = models['nitrogen'].predict(X_scaled)[0]
            st.metric("Nitrogen Rec.", str(pred_nit))
            
        with col3:
            st.subheader("🌱 Soil Health")
            pred_soil = models['soil'].predict(X_scaled)[0]
            st.metric("Soil Index", f"{pred_soil:.2f}")
            
            st.subheader("💧 Water Stress")
            pred_wat = models['water'].predict(X_scaled)[0]
            st.metric("Water Stress Level", str(pred_wat))
else:
    st.info("Waiting for models to finish training... Please check back in a minute and refresh the page.")
