"""
Exploratory Data Analysis Pipeline

This module analyses the processed water network
telemetry dataset before machine learning modelling.

"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__),"../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

# Create logger
logger = get_logger(__name__)

# Input dataset
INPUT_PATH = (
    "data/processed/"
    "clean_water_network_data.csv"
)

# Report paths
REPORT_PATH = (
    "reports/metrics/"
    "dataset_profile.txt"
)

SUMMARY_PATH = (
    "reports/metrics/"
    "summary_statistics.csv"
)

MISSING_PATH = (
    "reports/metrics/"
    "missing_values.csv"
)

FIGURE_PATH = (
    "reports/figures/"
)


def load_data():
    # Load processed dataset.
    try:

        logger.info(
            "Loading processed water telemetry dataset"
        )

        df = pd.read_csv(INPUT_PATH)

        logger.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )
        return df

    except Exception as error:

        logger.exception(
            f"Failed loading processed dataset: {error}"
        )
        raise


def check_missing_values(df):
    # Check missing values in dataset.
    logger.info("Checking missing values")

    missing_values = (
        df.isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )
    return missing_values


def generate_summary_statistics(df):

    # Generate dataset summary statistics.
    logger.info("Generating summary statistics")

    summary = (df.describe().transpose())
    return summary


def analyse_target_distribution(df):

    # Analyse failure prediction target distribution.
    logger.info("Analysing target distribution")

    target_count = (
        df["failure_within_30_days"]
        .value_counts()
    )
    return target_count


def create_feature_distribution(df, column_name):

    # Create feature distribution plot.
    try:
        logger.info(
            f"Creating distribution plot for {column_name}"
        )

        plt.figure(figsize=(8, 5))

        plt.hist(df[column_name].dropna(),bins=30)
        
        plt.title(f"{column_name} Distribution")

        plt.xlabel(column_name)

        plt.ylabel("Frequency")

        output_path = (
            FIGURE_PATH
            +
            f"{column_name}_distribution.png"
        )

        plt.savefig(output_path, bbox_inches="tight")

        plt.close()

        logger.info(
            f"Saved distribution plot: {output_path}"
        )

    except Exception as error:

        logger.exception(
            f"Failed creating distribution plot: {error}"
        )
        raise


def create_correlation_matrix(df):

    # Create correlation matrix visualisation.
    try:
        logger.info(
            "Creating correlation matrix"
        )

        correlation = (
            df.select_dtypes(
                include="number")
            .corr()
        )

        plt.figure(figsize=(12, 8))

        plt.imshow(correlation)

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title("Feature Correlation Matrix")

        output_path = (
            FIGURE_PATH
            +
            "correlation_heatmap.png")

        plt.savefig(output_path, bbox_inches="tight")

        plt.close()

        logger.info(
            f"Saved correlation matrix: {output_path}"
        )

    except Exception as error:

        logger.exception(
            f"Failed creating correlation matrix: {error}"
        )
        raise


def save_reports(df):

    # Save generated analysis reports.
    try:
        logger.info(
            "Saving analysis reports"
        )

        os.makedirs("reports/metrics", exist_ok=True)

        os.makedirs("reports/figures", exist_ok=True)

        summary = generate_summary_statistics(df)

        summary.to_csv(SUMMARY_PATH)

        missing = check_missing_values(df)

        missing.to_csv(MISSING_PATH)

        target = analyse_target_distribution(df)

        with open(REPORT_PATH,"w", encoding="utf-8") as file:

            file.write("Water Network Dataset Profile\n\n")
            
            file.write(f"Rows: {df.shape[0]}\n")

            file.write(f"Columns: {df.shape[1]}\n\n")

            file.write("Failure Distribution\n")

            file.write(str(target))

        logger.info(
            "Analysis reports saved successfully"
        )

    except Exception as error:
        
        logger.exception(
            f"Failed saving reports: {error}"
        )
        raise


def main():

    # Execute exploratory data analysis pipeline.
    try:
        
        logger.info(
            "Starting exploratory data analysis pipeline"
        )

        df = load_data()

        save_reports(df)

        features = [
            "pressure_bar",
            "flow_rate_lps",
            "temperature_c",
            "vibration_mm_s",
            "turbidity_ntu",
            "chlorine_mg_l",
            "pump_runtime_hours",
            "maintenance_age_days"
        ]

        for feature in features:
            create_feature_distribution(df,feature)

        create_correlation_matrix(df)

        logger.info(
            "Exploratory data analysis completed successfully"
        )

    except Exception as error:

        logger.exception(
            f"EDA pipeline failed: {error}"
        )
        raise

if __name__ == "__main__":

    main()