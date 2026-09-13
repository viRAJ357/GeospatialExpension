import pandas as pd
import numpy as np
import sys

def main():
    # Load the Kaggle crop recommendation dataset
    print("Loading real dataset...")
    url = 'https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv'
    try:
        real_df = pd.read_csv(url)
    except Exception as e:
        print(f"Error loading real dataset: {e}")
        sys.exit(1)

    # Load the original synthetic dataset to preserve structure
    print("Loading original dataset...")
    orig_df = pd.read_csv('Geospacial Data  .csv')

    print(f"Original shape: {orig_df.shape}, Real shape: {real_df.shape}")

    # Sample original dataset to match the size of real dataset
    if len(orig_df) < len(real_df):
        new_df = orig_df.sample(n=len(real_df), replace=True).reset_index(drop=True)
    else:
        new_df = orig_df.sample(n=len(real_df), replace=False).reset_index(drop=True)

    # Map the columns
    # Real dataset columns: N, P, K, temperature, humidity, ph, rainfall, label
    new_df['crop_type'] = real_df['label'].str.title()
    new_df['nitrogen_kg_ha'] = real_df['N']
    new_df['phosphorus_kg_ha'] = real_df['P']
    new_df['potassium_kg_ha'] = real_df['K']
    new_df['temperature_celsius'] = real_df['temperature']
    new_df['humidity_percent'] = real_df['humidity']
    new_df['soil_ph'] = real_df['ph']
    new_df['rainfall_mm'] = real_df['rainfall']

    # Overwrite the old csv
    print("Saving new dataset...")
    new_df.to_csv('Geospacial Data  .csv', index=False)
    print("Done! Real dataset adapted successfully.")

if __name__ == '__main__':
    main()
