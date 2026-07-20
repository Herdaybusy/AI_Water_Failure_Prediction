"""
Generate synthetic raw water network telemetry data.

This simulates data collected from:
- water pipes
- pumping stations
- IoT sensors
- maintenance systems

The generated dataset intentionally contains:
- missing values
- sensor anomalies
- outliers
- noisy measurements
- failure events
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Reproducibility
np.random.seed(42)


# Configuration
NUM_RECORDS = 120_000
NUM_ASSETS = 500

OUTPUT_PATH = "data/raw/water_network_telemetry.csv"


def generate_timestamps(n):
    """
    Generate hourly telemetry timestamps
    """
    start_date = datetime(2021, 1, 1)

    return [
        start_date + timedelta(hours=i)
        for i in range(n)
    ]


def generate_asset_ids(n):
    """
    Generate water infrastructure asset IDs
    """

    assets = [
        f"PIPE_{i:04d}"
        for i in range(1, NUM_ASSETS + 1)
    ]

    return np.random.choice(
        assets,
        size=n
    )


def generate_sensor_data(n):
    """
    Generate realistic water network measurements
    """

    data = pd.DataFrame({

        "timestamp": generate_timestamps(n),

        "asset_id": generate_asset_ids(n),

        # Normal operating ranges
        "pressure_bar": np.random.normal(
            loc=5,
            scale=0.8,
            size=n
        ),

        "flow_rate_lps": np.random.normal(
            loc=120,
            scale=25,
            size=n
        ),

        "temperature_c": np.random.normal(
            loc=14,
            scale=3,
            size=n
        ),

        "vibration_mm_s": np.random.normal(
            loc=2,
            scale=0.5,
            size=n
        ),

        "turbidity_ntu": np.random.normal(
            loc=1.5,
            scale=0.4,
            size=n
        ),

        "chlorine_mg_l": np.random.normal(
            loc=0.8,
            scale=0.15,
            size=n
        ),

        "pump_runtime_hours": np.random.randint(
            100,
            20000,
            size=n
        ),

        "maintenance_age_days": np.random.randint(
            1,
            1500,
            size=n
        )
    })

    return data


def introduce_failures(df):
    """
    Create realistic failure events.

    Only a small percentage of assets fail.
    """

    failure_probability = 0.03

    failures = np.random.choice(
        [0, 1],
        size=len(df),
        p=[
            1 - failure_probability,
            failure_probability
        ]
    )

    df["failure_within_30_days"] = failures


    # Failures create abnormal sensor behaviour

    failure_rows = df[
        df["failure_within_30_days"] == 1
    ].index


    df.loc[
        failure_rows,
        "pressure_bar"
    ] *= np.random.uniform(
        0.3,
        0.7,
        len(failure_rows)
    )


    df.loc[
        failure_rows,
        "vibration_mm_s"
    ] *= np.random.uniform(
        2,
        5,
        len(failure_rows)
    )


    df.loc[
        failure_rows,
        "turbidity_ntu"
    ] *= np.random.uniform(
        2,
        5,
        len(failure_rows)
    )


    return df



def introduce_missing_values(df):
    """
    Simulate broken sensors
    """

    missing_rate = 0.02

    sensor_columns = [
        "pressure_bar",
        "flow_rate_lps",
        "temperature_c",
        "vibration_mm_s",
        "turbidity_ntu",
        "chlorine_mg_l"
    ]


    for col in sensor_columns:

        missing_indices = np.random.choice(
            df.index,
            size=int(len(df) * missing_rate),
            replace=False
        )

        df.loc[
            missing_indices,
            col
        ] = np.nan


    return df



def introduce_outliers(df):
    """
    Simulate sensor spikes
    """

    outlier_count = int(len(df) * 0.01)


    indices = np.random.choice(
        df.index,
        size=outlier_count,
        replace=False
    )


    df.loc[
        indices,
        "pressure_bar"
    ] = np.random.uniform(
        10,
        15,
        outlier_count
    )


    df.loc[
        indices,
        "flow_rate_lps"
    ] = np.random.uniform(
        300,
        600,
        outlier_count
    )


    return df



def main():

    print("Generating raw telemetry dataset...")


    df = generate_sensor_data(
        NUM_RECORDS
    )


    df = introduce_failures(df)

    df = introduce_missing_values(df)

    df = introduce_outliers(df)


    # Add leak indicator
    df["leak_detected"] = (
        (
            df["pressure_bar"] < 3
        )
        |
        (
            df["turbidity_ntu"] > 4
        )
    ).astype(int)


    # Create output directory
    os.makedirs(
        "data/raw",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print(
        f"Dataset created: {OUTPUT_PATH}"
    )

    print(
        df.head()
    )

    print(
        "\nDataset shape:",
        df.shape
    )


if __name__ == "__main__":
    main()