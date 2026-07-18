"""
Raw Data Quality Profiling Pipeline

This module analyses incoming water network telemetry
before processing through the ETL pipeline.

"""

import os
import pandas as pd
import sys

# project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__),"../.."))


sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger  

# Create logger for this module
logger = get_logger(__name__)

# Input dataset
INPUT_PATH = ( "data/raw/water_network_telemetry.csv")

# Quality report output
REPORT_PATH = ("reports/raw_data_quality_report.txt")

def load_data():
    """
    Load raw telemetry dataset.
    """
    try:

        logger.info("Loading raw telemetry dataset" )

        df = pd.read_csv(INPUT_PATH)

        logger.info( f"Dataset loaded successfully. Shape: {df.shape}" )
        return df

    except Exception as error:

        logger.exception( f"Failed loading raw dataset: {error}")
        raise

def check_missing_values(df):
    """
    Identify missing values in each column.
    """

    logger.info( "Checking missing values")

    missing = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return missing


def check_duplicates(df):
    """
    Count duplicate records.
    """
    logger.info( "Checking duplicate records")

    duplicates = ( df.duplicated().sum() )
    return duplicates


def check_data_types(df):
    """
    Review column data types.
    """

    logger.info( "Checking data types")
    return df.dtypes


def check_sensor_ranges(df):
    """
    Detect unrealistic sensor readings.
    """
    logger.info("Checking sensor value ranges")

    anomalies = {}

    # Pressure should be positive

    anomalies["pressure_bar_invalid"] = (df[ "pressure_bar"] < 0 ).sum()

    # Flow cannot be negative

    anomalies["flow_rate_invalid"] = (df["flow_rate_lps"] < 0).sum()

    # Vibration cannot be negative

    anomalies["vibration_invalid"] = (df[ "vibration_mm_s" ] < 0).sum()

    # Turbidity cannot be negative

    anomalies["turbidity_invalid"] = ( df["turbidity_ntu"] < 0 ).sum()

    return anomalies


def generate_report(df):
    """
    Generate complete quality report.
    """

    logger.info("Generating data quality report")


    missing = check_missing_values(df)

    duplicates = check_duplicates(df)
    
    data_types = check_data_types(df)

    anomalies = check_sensor_ranges(df)

    report = f"""

Water Network Telemetry Data Quality Report


Dataset Information
-------------------

Rows:
{df.shape[0]}


Columns:
{df.shape[1]}


Missing Values
--------------

{missing}


Duplicate Records
-----------------

{duplicates}


Data Types
----------

{data_types}


Sensor Validation
-----------------

{anomalies}

"""

    return report



def save_report(report):
    """
    Save quality report to file.
    """

    try:

        os.makedirs(

            "reports",

            exist_ok=True

        )

        with open(

            REPORT_PATH, "w"

        ) as file:

            file.write(
                report
            )


        logger.info( f"Quality report saved successfully: {REPORT_PATH}")


    except Exception as error:

        logger.exception( f"Failed saving quality report: {error}")

        raise


def main():

    """
    Execute data quality pipeline.
    """

    try:

        logger.info("Starting data quality profiling pipeline")

        df = load_data()

        report = generate_report(df)

        save_report(report)

        logger.info("Data quality profiling completed successfully")

    except Exception as error:


        logger.exception( f"Data quality pipeline failed: {error}" )

        raise

if __name__ == "__main__":

    main()