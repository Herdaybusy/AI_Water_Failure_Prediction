"""
Machine Learning Training Pipeline

This module trains a machine learning model
to predict water network failures.

"""


import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib

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

# Model path
MODEL_PATH = (
    "models/"
    "water_failure_model.pkl"
)

# Metrics path
METRIC_PATH = (
    "reports/metrics/"
    "model_metrics.txt"
)


def load_data():
    # Load processed dataset.
    try:
        logger.info(
            "Loading processed dataset for model training"
        )

        df = pd.read_csv(INPUT_PATH)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as error:

        logger.exception(f"Failed loading training dataset: {error}")
        raise



def prepare_features(df):
    # Prepare features and target variable.
    try:
        logger.info("Preparing training features")

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

        y = df["failure_within_30_days"]

        logger.info(
            f"Feature preparation completed. Features: {X.shape[1]}"
        )
        return X, y

    except Exception as error:

        logger.exception(f"Failed preparing features: {error}")
        raise



def split_data(X, y):
    # Split data into training and testing sets.
    try:
        logger.info(
            "Splitting dataset into training and testing sets"
        )

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        logger.info(f"Training data shape: {X_train.shape}")

        logger.info(f"Testing data shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as error:

        logger.exception(f"Failed splitting dataset: {error}")
        raise


def train_model(X_train, y_train):
    # Train machine learning model.
    try:
        logger.info("Training Random Forest model")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        logger.info(
            "Model training completed successfully"
        )
        return model

    except Exception as error:

        logger.exception(
            f"Failed training model: {error}"
        )
        raise


def evaluate_model(model, X_test, y_test):
    # Evaluate model performance.
    try:
        logger.info(
            "Evaluating model performance"
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:,1]

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "precision": precision_score(
                y_test,
                predictions
            ),

            "recall": recall_score(
                y_test,
                predictions
            ),

            "f1_score": f1_score(
                y_test,
                predictions
            ),

            "roc_auc": roc_auc_score(
                y_test,
                probabilities
            )

        }

        logger.info(
            f"Model evaluation completed: {metrics}"
        )
        return metrics

    except Exception as error:

        logger.exception(
            f"Failed evaluating model: {error}"
        )
        raise


def save_model(model):
    # Save trained model.
    try:
        logger.info(
            "Saving trained model"
        )

        os.makedirs("models",exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        logger.info(
            f"Model saved successfully: {MODEL_PATH}"
        )

    except Exception as error:

        logger.exception(
            f"Failed saving model: {error}"
        )
        raise


def save_metrics(metrics):
    # Save model evaluation metrics.
    try:
        logger.info("Saving model metrics")

        os.makedirs("reports/metrics", exist_ok=True)

        with open(
            METRIC_PATH,
            "w"
        ) as file:

            file.write(
                "Water Failure Prediction Model Metrics\n\n"
            )

            for metric, value in metrics.items():

                file.write(
                    f"{metric}: {value}\n"
                )

        logger.info(
            "Model metrics saved successfully"
        )

    except Exception as error:

        logger.exception(
            f"Failed saving metrics: {error}"
        )
        raise


def main():
    # Execute machine learning training pipeline.
    try:
        logger.info(
            "Starting machine learning training pipeline"
        )

        df = load_data()

        X, y = prepare_features(df)

        X_train, X_test, y_train, y_test = split_data(
            X,
            y
        )

        model = train_model(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        save_model(model)

        save_metrics(metrics)

        logger.info(
            "Machine learning training pipeline completed successfully"
        )

    except Exception as error:

        logger.exception(
            f"ML training pipeline failed: {error}"
        )
        raise

if __name__ == "__main__":
    main()