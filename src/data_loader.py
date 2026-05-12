import kagglehub
import pandas as pd
import os

def prepare_solar_dataset():
    # 1. Download from Kaggle
    dataset_path = kagglehub.dataset_download("anikannal/solar-power-generation-data")
    print(f"Dataset securely downloaded to: {dataset_path}")

    # 2. Map exact paths 
    generation_path = os.path.join(dataset_path, "Plant_1_Generation_Data.csv")
    weather_sensor_path = os.path.join(dataset_path, "Plant_1_Weather_Sensor_Data.csv")

    # 3. Load raw data
    generation_dataframe = pd.read_csv(generation_path)
    weather_dataframe = pd.read_csv(weather_sensor_path)

    # 4. Standardize Date/Time
    generation_dataframe['DATE_TIME'] = pd.to_datetime(generation_dataframe['DATE_TIME'])
    weather_dataframe['DATE_TIME'] = pd.to_datetime(weather_dataframe['DATE_TIME'])

    # 5. Merge datasets securely
    solar_dataframe = pd.merge(generation_dataframe, weather_dataframe, on='DATE_TIME', how='inner')

    # 6. Feature Engineering
    solar_dataframe['hour_of_day'] = solar_dataframe['DATE_TIME'].dt.hour
    
    # Filter out nighttime data (zeros skew the regression models)
    solar_dataframe = solar_dataframe[solar_dataframe['IRRADIATION'] > 0]

    # 7. Export locally
    os.makedirs('data', exist_ok=True)
    processed_path = 'data/processed_solar_data.csv'
    solar_dataframe.to_csv(processed_path, index=False)
    print(f"Data pipeline complete. Cleaned file saved to {processed_path}")

if __name__ == "__main__":
    prepare_solar_dataset()