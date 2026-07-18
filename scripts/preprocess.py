import pandas as pd
import os

df = pd.read_csv(
    "data/raw/raw_sensor_data.csv"
)

df["PressureVariance"] = (
    df["Pressure"] - 55
).abs()

df["AssetHealthScore"] = (
    100
    - (df["AssetAge"] * 1.5)
    - (df["DaysSinceMaintenance"] * 0.05)
)

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    "data/processed/processed_sensor_data.csv",
    index=False
)

print("Feature engineering complete.")