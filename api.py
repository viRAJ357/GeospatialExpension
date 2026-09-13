from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load Models
try:
    models = {
        'yield': joblib.load('crop_yield_prediction_model.pkl'),
        'disease': joblib.load('disease_detection_model.pkl'),
        'irrigation': joblib.load('irrigation_recommendation_model.pkl'),
        'nitrogen': joblib.load('nitrogen_recommendation_model.pkl'),
        'soil': joblib.load('soil_health_model.pkl'),
        'water': joblib.load('water_stress_model.pkl')
    }
    feature_cols = joblib.load('feature_columns.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    print("✅ All models and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract inputs
        crop_type = data.get('crop_type', 'Rice')
        N = float(data.get('nitrogen', 50))
        P = float(data.get('phosphorus', 50))
        K = float(data.get('potassium', 50))
        temp = float(data.get('temperature', 25))
        hum = float(data.get('humidity', 60))
        ph = float(data.get('ph', 6.5))
        rain = float(data.get('rainfall', 100))

        # 1. Synthesize derived features
        n_factor = (N - 0) / 140
        rain_factor = min(rain / 300, 1.0)
        ndvi = np.clip(0.2 + (n_factor * 0.4) + (rain_factor * 0.3), 0.1, 0.9)
        soil_moisture_percent = np.clip((rain / 300) * 80 - (temp / 40) * 20 + 20, 10, 90)
        npk_score = (N + P + K) / 400 * 50
        ph_score = max(0, 1 - abs(ph - 6.5) / 3.5) * 50
        soil_fertility_index = np.clip(npk_score + ph_score, 10, 100)
        
        # Engineered features
        npk_total = N + P + K
        n_to_p_ratio = N / (P + 1)
        temp_humidity_index = (temp * hum) / 100
        water_availability_index = (soil_moisture_percent / 100 + rain_factor) / 2
        ph_score_health = 1 - abs(ph - 6.75) / 6.75
        ph_score_health = max(0, min(ph_score_health, 1))
        soil_health_score = (soil_fertility_index / 100 + ph_score_health) / 2

        # Encode crop type
        crop_type_encoded = label_encoders['crop_type'].transform([crop_type])[0]

        # First, we need to create a dummy dataframe for scaling, containing all scaler columns
        # To scale properly, we must match the scaler's expected input columns
        # Let's inspect scaler features (it expects a certain order)
        scaler_cols = scaler.feature_names_in_
        
        raw_data = {
            'ndvi': ndvi,
            'soil_moisture_percent': soil_moisture_percent,
            'temperature_celsius': temp,
            'humidity_percent': hum,
            'rainfall_mm': rain,
            'nitrogen_kg_ha': N,
            'phosphorus_kg_ha': P,
            'potassium_kg_ha': K,
            'ph_level': ph,
            'soil_fertility_index': soil_fertility_index,
            'npk_total': npk_total,
            'n_to_p_ratio': n_to_p_ratio,
            'n_to_k_ratio': N / (K + 1),
            'p_to_k_ratio': P / (K + 1),
            'temp_humidity_index': temp_humidity_index,
            'water_availability_index': water_availability_index,
            'soil_health_score': soil_health_score,
            'latitude': 20.0,
            'longitude': 78.0
        }
        
        df_raw = pd.DataFrame([raw_data])
        # Ensure correct order
        df_raw = df_raw[scaler_cols]
        scaled_features = scaler.transform(df_raw)
        df_scaled = pd.DataFrame(scaled_features, columns=scaler_cols)
        
        # Add non-scaled features needed for prediction
        df_scaled['crop_type_encoded'] = crop_type_encoded
        # Some targets are needed as inputs for other models, so we run them sequentially
        
        # 1. Soil Health
        soil_pred = models['soil'].predict(df_scaled[feature_cols['soil_features']])[0]
        
        # 2. Water Stress
        water_pred_enc = models['water'].predict(df_scaled[feature_cols['water_features']])[0]
        df_scaled['water_stress_level_encoded'] = water_pred_enc
        water_stress_label = label_encoders['water_stress_level'].inverse_transform([water_pred_enc])[0]
        
        # 3. Disease Detection
        disease_pred_enc = models['disease'].predict(df_scaled[feature_cols['disease_features']])[0]
        df_scaled['disease_name_encoded'] = disease_pred_enc
        disease_label = label_encoders['disease_name'].inverse_transform([disease_pred_enc])[0]
        
        # 4. Irrigation
        irrigation_pred_enc = models['irrigation'].predict(df_scaled[feature_cols['irrigation_features']])[0]
        df_scaled['irrigation_needed_encoded'] = irrigation_pred_enc
        irrigation_label = label_encoders['irrigation_needed'].inverse_transform([irrigation_pred_enc])[0]
        
        # 5. Crop Yield (needs all previous)
        # We need to temporarily set crop_yield to 0 since nitrogen needs it, but wait!
        # Nitrogen feature needs crop_yield. Let's predict crop yield first!
        yield_pred = models['yield'].predict(df_scaled[feature_cols['yield_features']])[0]
        df_scaled['crop_yield'] = yield_pred
        
        # 6. Nitrogen Recommendation
        nit_pred = models['nitrogen'].predict(df_scaled[feature_cols['nitrogen_features']])[0]
        
        return jsonify({
            'success': True,
            'predictions': {
                'soil_fertility_index': float(soil_pred),
                'water_stress_level': water_stress_label,
                'disease_name': disease_label,
                'irrigation_needed': irrigation_label,
                'crop_yield_tons_ha': float(yield_pred),
                'nitrogen_recommendation_kg_ha': float(nit_pred),
                'derived_ndvi': float(ndvi)
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
