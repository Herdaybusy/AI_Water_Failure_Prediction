"""
Failure Prediction Pipeline

This module loads the trained machine learning model
and generates failure predictions for new water
network telemetry data.

"""

import os
import sys
import pandas as pd
import joblib

# project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(
        os.path.dirname(__file__),"../.."))

sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger

# Create logger
logger = get_logger(__name__)


# Input data
INPUT_PATH = (
    "data/"
    "new_sensor_readings.csv"
)

# Model path
MODEL_PATH = (
    "models/"
    "water_failure_model.pkl"
)

# Prediction output
OUTPUT_PATH = (
    "results/"
    "predictions.csv"
)


def load_model():
    # Load trained machine learning model.
    try:
        logger.info(
            "Loading trained prediction model"
        )

        model = joblib.load(
            MODEL_PATH
        )

        logger.info(
            "Prediction model loaded successfully"
        )

        return model

    except Exception as error:

        logger.exception(
            f"Failed loading prediction model: {error}"
        )
        raise



def load_prediction_data():
    # Load new sensor readings.
    try:
        logger.info(
            "Loading new sensor telemetry data"
        )

        df = pd.read_csv(
            INPUT_PATH
        )

        logger.info(
            f"Prediction dataset loaded successfully. Shape: {df.shape}"
        )
        return df

    except Exception as error:

        logger.exception(
            f"Failed loading prediction dataset: {error}"
        )

        raise


def prepare_features(df):
    # Prepare prediction features.
    try:
        logger.info(
            "Preparing prediction features"
        )

        features = [
            "pressure_bar",
            "flow_rate_lps",
            "temperature_c",
            "vibration_mm_s",
            "turbidity_ntu",
            "chlorine_mg_l",
            "pump_runtime_hours",
            "maintenance_age_days",
            "maintenance_risk_score",
            "pump_utilisation_score",
            "pressure_health_score",
            "flow_efficiency",
            "operational_risk_score"
        ]

        X = df[features]

        logger.info(
            f"Prediction features prepared. Features: {X.shape[1]}"
        )
        return X

    except Exception as error:

        logger.exception(
            f"Failed preparing prediction features: {error}"
        )
        raise


def classify_risk(probability):
    # Classify failure risk level.
    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.40:
        return "MEDIUM"

    else:
        return "LOW"


def generate_recommendation(risk):
    # Generate maintenance recommendation.
    recommendations = {
        "HIGH":
        "Immediate inspection and preventive maintenance required",

        "MEDIUM":
        "Schedule inspection and monitor asset condition",

        "LOW":
        "Continue normal operation and routine monitoring"
    }

    return recommendations[risk]


def generate_predictions(model, df, X):
    # Generate failure predictions.
    try:
        logger.info(
            "Generating failure predictions"
        )
        
        probabilities = (
            model.predict_proba(X)[:,1]
        )

        df["failure_probability"] = probabilities

        df["risk_level"] = (
            df["failure_probability"]
            .apply(classify_risk)
        )

        df["recommendation"] = (
            df["risk_level"]
            .apply(generate_recommendation)
        )

        logger.info(
            "Failure predictions generated successfully"
        )
        return df

    except Exception as error:

        logger.exception(
            f"Failed generating predictions: {error}"
        )
        raise


def save_predictions(df):
    # Save prediction results.
    try:
        logger.info(
            "Saving prediction results"
        )

        os.makedirs(

            "results",

            exist_ok=True
        )

        df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        logger.info(
            f"Prediction results saved successfully: {OUTPUT_PATH}"
        )
        
    except Exception as error:

        logger.exception(
            f"Failed saving predictions: {error}"
        )
        raise


def main():
    # Execute failure prediction pipeline.
    try:
        logger.info(
            "Starting failure prediction pipeline"
        )

        model = load_model()

        df = load_prediction_data()

        X = prepare_features(
            df
        )

        predictions = generate_predictions(

            model,

            df,

            X
        )

        save_predictions(
            predictions
        )

        logger.info(
            "Failure prediction pipeline completed successfully"
        )

    except Exception as error:

        logger.exception(
            f"Prediction pipeline failed: {error}"
        )
        raise


if __name__ == "__main__":

    main()