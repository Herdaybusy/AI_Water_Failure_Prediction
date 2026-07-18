"""
Water Network Data Cleaning ETL Pipeline

Steps:

- Load raw data
- Remove duplicates
- Handle missing values
- Validate sensor readings
- Create new features
- Save processed data

"""

import os
import sys
import pandas as pd
import numpy as np

# project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__),"../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

# Create logger for this module
logger = get_logger(__name__)

# Input dataset
INPUT_PATH = ("data/raw/""water_network_telemetry.csv")

# Output dataset
OUTPUT_PATH = ("data/processed/" "clean_water_network_data.csv")


def load_data():
    
    # Load raw telemetry dataset.
    try:
        logger.info("Loading raw water telemetry dataset")

        df = pd.read_csv(INPUT_PATH)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as error:

        logger.exception(f"Failed loading raw dataset: {error}")
        raise


def remove_duplicates(df):


    # Remove duplicate records.

    logger.info("Checking duplicate records")

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    removed = before - after

    logger.info(f"Removed {removed} duplicate records")
    return df



def handle_missing_values(df):
    
    # Handle missing numerical values.
    logger.info("Handling missing values")

    numeric_columns = (df.select_dtypes(
        include=np.number
        ).columns
    )


    missing_values = (
        df[numeric_columns]
        .isnull()
        .sum()
        .sum()
    )


    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(
            df[numeric_columns]
            .median()
        )
    )

    logger.info(f"Filled {missing_values} missing values")
    return df


def validate_sensor_data(df):
    
    # Validate sensor readings.
    logger.info("Validating sensor measurements")

    invalid_pressure = (df["pressure_bar"] < 0).sum()

    invalid_flow = (df["flow_rate_lps"] < 0).sum()

    invalid_vibration = (df["vibration_mm_s"] < 0).sum()

    logger.info(f"Invalid pressure readings: {invalid_pressure}")
    
    logger.info(f"Invalid flow readings: {invalid_flow}")

    logger.info(f"Invalid vibration readings: {invalid_vibration}")

    # Replace invalid values with missing values
    df.loc[
        df["pressure_bar"] < 0,
        "pressure_bar"
    ] = np.nan

    df.loc[
        df["flow_rate_lps"] < 0,
        "flow_rate_lps"
    ] = np.nan

    df.loc[
        df["vibration_mm_s"] < 0,
        "vibration_mm_s"
    ] = np.nan

    return df


def create_features(df):

    # Create additional machine learning features.

    logger.info("Creating new features")

     # Asset maintenance risk score

    df["maintenance_risk_score"] = (

        df["maintenance_age_days"]
        /
        df["maintenance_age_days"].max()

    )



    # Pump utilisation indicator

    df["pump_utilisation_score"] = (

        df["pump_runtime_hours"]
        /
        df["pump_runtime_hours"].max()

    )



    # Pressure stability indicator

    df["pressure_health_score"] = (

        df["pressure_bar"]

        /

        df["pressure_bar"].mean()

    )



    # Flow efficiency indicator

    df["flow_efficiency"] = (

        df["flow_rate_lps"]

        /

        (df["pressure_bar"] + 0.1)

    )



    # Overall operational risk indicator

    df["operational_risk_score"] = (

        df["maintenance_risk_score"]

        +

        df["pump_utilisation_score"]

        +

        (df["vibration_mm_s"] / df["vibration_mm_s"].max())

    )

    logger.info("Feature creation completed")
    return df



def save_data(df):

    # Save processed dataset.
    try:

        logger.info("Saving processed dataset")

        os.makedirs("data/processed", exist_ok=True)

        df.to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Processed dataset saved successfully: {OUTPUT_PATH}")

    except Exception as error:

        logger.exception(f"Failed saving processed dataset: {error}")
        raise


def main():

    # Execute ETL pipeline.
    try:
        
        logger.info("Starting ETL cleaning pipeline")

        df = load_data()

        df = remove_duplicates(df)

        df = handle_missing_values(df)
        
        df = validate_sensor_data(df)

        df = handle_missing_values(df)

        df = create_features(df)

        save_data(df)

        logger.info("ETL pipeline completed successfully")

    except Exception as error:

        logger.exception(f"ETL pipeline failed: {error}")
        raise

if __name__ == "__main__":

    main()