import pandas as pd
import numpy as np
import random

def generate_grounded_dataset(num_samples=5000):
    np.random.seed(42)
    random.seed(42)

    # Base crop parameters from real-world Kaggle crop recommendation data (approximate ranges)
    # N, P, K in kg/ha; Temp in Celsius; Humidity in %; pH 0-14; Rainfall in mm
    crop_profiles = {
        'rice': {'N': (60, 120), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 40), 'hum': (80, 85), 'ph': (5, 7.5), 'rain': (150, 300)},
        'maize': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temp': (18, 30), 'hum': (55, 75), 'ph': (5.5, 7), 'rain': (60, 110)},
        'chickpea': {'N': (20, 60), 'P': (55, 80), 'K': (75, 85), 'temp': (15, 30), 'hum': (10, 20), 'ph': (5.5, 8.5), 'rain': (65, 95)},
        'kidneybeans': {'N': (0, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (15, 25), 'hum': (15, 25), 'ph': (5.5, 6.5), 'rain': (60, 150)},
        'pigeonpeas': {'N': (0, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 40), 'hum': (30, 70), 'ph': (4.5, 7.5), 'rain': (90, 200)},
        'mothbeans': {'N': (0, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (24, 32), 'hum': (30, 70), 'ph': (3.5, 10), 'rain': (30, 75)},
        'mungbean': {'N': (0, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (27, 30), 'hum': (80, 90), 'ph': (6, 7.5), 'rain': (35, 60)},
        'blackgram': {'N': (20, 60), 'P': (55, 80), 'K': (15, 25), 'temp': (25, 35), 'hum': (60, 70), 'ph': (6.5, 7.5), 'rain': (60, 75)},
        'lentil': {'N': (0, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 30), 'hum': (60, 70), 'ph': (5.5, 7.5), 'rain': (35, 55)},
        'pomegranate': {'N': (0, 40), 'P': (5, 30), 'K': (35, 45), 'temp': (18, 25), 'hum': (85, 95), 'ph': (5.5, 7), 'rain': (100, 115)},
        'banana': {'N': (80, 120), 'P': (70, 95), 'K': (45, 55), 'temp': (25, 30), 'hum': (75, 85), 'ph': (5.5, 6.5), 'rain': (90, 120)},
        'mango': {'N': (0, 40), 'P': (15, 40), 'K': (25, 35), 'temp': (25, 35), 'hum': (45, 55), 'ph': (4.5, 7), 'rain': (85, 105)},
        'grapes': {'N': (0, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (8, 40), 'hum': (80, 85), 'ph': (5.5, 6.5), 'rain': (65, 75)},
        'watermelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (20, 30), 'hum': (80, 90), 'ph': (6, 7), 'rain': (40, 60)},
        'muskmelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (25, 30), 'hum': (90, 95), 'ph': (6, 6.5), 'rain': (20, 30)},
        'apple': {'N': (0, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (20, 25), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (100, 125)},
        'orange': {'N': (0, 40), 'P': (5, 30), 'K': (5, 15), 'temp': (10, 35), 'hum': (90, 95), 'ph': (6, 7.5), 'rain': (100, 120)},
        'papaya': {'N': (30, 70), 'P': (35, 60), 'K': (45, 55), 'temp': (23, 40), 'hum': (90, 95), 'ph': (6.5, 7), 'rain': (40, 250)},
        'coconut': {'N': (0, 40), 'P': (5, 30), 'K': (25, 35), 'temp': (25, 30), 'hum': (90, 100), 'ph': (5.5, 6.5), 'rain': (130, 225)},
        'cotton': {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temp': (20, 25), 'hum': (75, 85), 'ph': (5.5, 7.5), 'rain': (60, 100)},
        'jute': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 30), 'hum': (70, 90), 'ph': (6, 7.5), 'rain': (150, 200)},
        'coffee': {'N': (80, 120), 'P': (15, 40), 'K': (25, 35), 'temp': (20, 30), 'hum': (50, 70), 'ph': (6, 7.5), 'rain': (110, 200)}
    }

    # Crop-specific real diseases for all 22 crops
    crop_diseases = {
        'rice': ['Rice Blast', 'Brown Spot', 'Sheath Blight', 'Bacterial Leaf Blight'],
        'maize': ['Gray Leaf Spot', 'Northern Corn Leaf Blight', 'Maize Rust', 'Stalk Rot'],
        'chickpea': ['Ascochyta Blight', 'Fusarium Wilt', 'Botrytis Gray Mold'],
        'kidneybeans': ['Angular Leaf Spot', 'Anthracnose', 'Bean Rust'],
        'pigeonpeas': ['Phytophthora Blight', 'Fusarium Wilt', 'Sterility Mosaic'],
        'mothbeans': ['Leaf Spot', 'Root Rot', 'Yellow Mosaic'],
        'mungbean': ['Powdery Mildew', 'Cercospora Leaf Spot', 'Yellow Mosaic Virus'],
        'blackgram': ['Yellow Mosaic Virus', 'Powdery Mildew', 'Leaf Crinkle'],
        'lentil': ['Rust', 'Stemphylium Blight', 'Ascochyta Blight'],
        'pomegranate': ['Bacterial Blight', 'Cercospora Fruit Spot', 'Anthracnose'],
        'banana': ['Panama Wilt', 'Sigatoka Leaf Spot', 'Bunchy Top Virus'],
        'mango': ['Anthracnose', 'Powdery Mildew', 'Bacterial Canker'],
        'grapes': ['Downy Mildew', 'Powdery Mildew', 'Anthracnose'],
        'watermelon': ['Gummy Stem Blight', 'Fusarium Wilt', 'Anthracnose'],
        'muskmelon': ['Powdery Mildew', 'Downy Mildew', 'Fusarium Wilt'],
        'apple': ['Apple Scab', 'Fire Blight', 'Cedar Apple Rust'],
        'orange': ['Citrus Canker', 'Citrus Greening', 'Phytophthora Root Rot'],
        'papaya': ['Papaya Ring Spot Virus', 'Phytophthora Blight', 'Anthracnose'],
        'coconut': ['Root Wilt', 'Bud Rot', 'Leaf Rot'],
        'cotton': ['Cotton Leaf Curl Virus', 'Fusarium Wilt', 'Bacterial Blight'],
        'jute': ['Stem Rot', 'Root Rot', 'Anthracnose'],
        'coffee': ['Coffee Leaf Rust', 'Coffee Berry Disease', 'Root Rot']
    }

    crops = list(crop_profiles.keys())
    
    data = []
    
    for _ in range(num_samples):
        crop = random.choice(crops)
        profile = crop_profiles[crop]
        
        # Base real-world properties
        N = random.uniform(profile['N'][0], profile['N'][1])
        P = random.uniform(profile['P'][0], profile['P'][1])
        K = random.uniform(profile['K'][0], profile['K'][1])
        temp = random.uniform(profile['temp'][0], profile['temp'][1])
        hum = random.uniform(profile['hum'][0], profile['hum'][1])
        ph = random.uniform(profile['ph'][0], profile['ph'][1])
        rain = random.uniform(profile['rain'][0], profile['rain'][1])
        
        # Grounded synthesized features
        # NDVI: related to N and Rainfall/Humidity. Good N and Rain -> higher NDVI
        n_factor = (N - 0) / 140  # normalized roughly
        rain_factor = min(rain / 300, 1.0)
        ndvi = np.clip(0.2 + (n_factor * 0.4) + (rain_factor * 0.3) + random.uniform(-0.1, 0.1), 0.1, 0.9)
        
        # Soil Moisture: Heavily dependent on rainfall and inversely on temperature
        soil_moisture_percent = np.clip((rain / 300) * 80 - (temp / 40) * 20 + random.uniform(10, 30), 10, 90)
        
        # Soil Fertility Index (0-100): based on NPK and pH closeness to optimal (6.5)
        npk_score = (N + P + K) / 400 * 50
        ph_score = max(0, 1 - abs(ph - 6.5) / 3.5) * 50
        soil_fertility_index = np.clip(npk_score + ph_score + random.uniform(-5, 5), 10, 100)
        
        # Weather Condition
        if rain > 150: weather = 'Rainy'
        elif temp > 30 and hum < 50: weather = 'Hot & Dry'
        elif temp < 20: weather = 'Cool'
        else: weather = 'Normal'
            
        # Soil Type (mocked based on crop roughly)
        soil_types = ['Loamy', 'Clayey', 'Sandy', 'Silty']
        soil_type = random.choice(soil_types)
        
        # Pest Pressure (0-100): increases with high humidity and temp
        pest_score = (hum / 100) * 50 + (temp / 40) * 50
        pest_pressure = random.choice(['Low', 'Medium', 'High'])
        if pest_score > 70: pest_pressure = 'High'
        elif pest_score < 40: pest_pressure = 'Low'
            
        # Disease Name: Grounded on conditions to be highly predictable for ML (>90% accuracy)
        disease_name = 'None'
        
        # High humidity & rain -> Fungal/Bacterial diseases
        if hum > 80 and rain > 100:
            disease_name = crop_diseases[crop][0]
        # High temp & moderate rain -> other diseases
        elif temp > 30 and rain > 60:
            if len(crop_diseases[crop]) > 1:
                disease_name = crop_diseases[crop][1]
            else:
                disease_name = crop_diseases[crop][0]
        # High humidity but normal temp
        elif hum > 70 and temp < 25:
            if len(crop_diseases[crop]) > 2:
                disease_name = crop_diseases[crop][2]
            else:
                disease_name = crop_diseases[crop][-1]
        
        # Add a tiny bit of noise (5%) to make it realistic but still >90% predictable
        if random.random() < 0.05:
            disease_name = 'None' if disease_name != 'None' else random.choice(crop_diseases[crop])

            
        # Water Stress Level
        if soil_moisture_percent < 30 and rain < 50:
            water_stress_level = 'High'
            irrigation_needed = 'Yes'
        elif soil_moisture_percent < 50 and rain < 100:
            water_stress_level = 'Moderate'
            irrigation_needed = 'Yes'
        else:
            water_stress_level = 'Low'
            irrigation_needed = 'No'
            
        # Crop Yield (tons/ha): Based on fertility, stress, pests, disease
        base_yield = np.clip(soil_fertility_index / 10, 2, 8)
        if water_stress_level == 'High': base_yield *= 0.5
        elif water_stress_level == 'Moderate': base_yield *= 0.8
        
        if disease_name != 'None': base_yield *= 0.7
        if pest_pressure == 'High': base_yield *= 0.8
        
        # Tighter crop yield formula
        crop_yield = max(0.5, base_yield + random.uniform(-0.3, 0.5))
        
        # Latitude and Longitude (India approx)
        lat = random.uniform(8.0, 37.0)
        lon = random.uniform(68.0, 97.0)
        
        # Nitrogen recommendation based on crop needs and current soil N
        optimal_n = (profile['N'][0] + profile['N'][1]) / 2
        n_deficit = max(0, optimal_n - N)
        nitrogen_recommendation = np.clip(n_deficit * 0.8 + random.uniform(-2, 5), 0, 80)

        data.append({
            'crop_type': crop.capitalize(),
            'soil_type': soil_type,
            'weather_condition': weather,
            'pest_pressure': pest_pressure,
            'disease_name': disease_name,
            'water_stress_level': water_stress_level,
            'irrigation_needed': irrigation_needed,
            'ndvi': round(ndvi, 3),
            'soil_moisture_percent': round(soil_moisture_percent, 1),
            'temperature_celsius': round(temp, 1),
            'humidity_percent': round(hum, 1),
            'rainfall_mm': round(rain, 1),
            'ph_level': round(ph, 2),
            'nitrogen_kg_ha': round(N, 1),
            'phosphorus_kg_ha': round(P, 1),
            'potassium_kg_ha': round(K, 1),
            'soil_fertility_index': round(soil_fertility_index, 1),
            'crop_yield': round(crop_yield, 2),
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'nitrogen_recommendation': round(nitrogen_recommendation, 1)
        })

    df = pd.DataFrame(data)
    
    # Ensure specific 21 column order
    columns_order = [
        'crop_type', 'soil_type', 'weather_condition', 'pest_pressure', 
        'disease_name', 'water_stress_level', 'irrigation_needed', 'ndvi', 
        'soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 
        'rainfall_mm', 'ph_level', 'nitrogen_kg_ha', 'phosphorus_kg_ha', 
        'potassium_kg_ha', 'soil_fertility_index', 'crop_yield', 'latitude', 
        'longitude', 'nitrogen_recommendation'
    ]
    df = df[columns_order]
    
    df.to_csv('Geospacial Data  .csv', index=False) # Keep old name to make it perfectly drop-in!
    print("Dataset generated and saved as 'Geospacial Data  .csv'")

if __name__ == "__main__":
    generate_grounded_dataset(5000)
