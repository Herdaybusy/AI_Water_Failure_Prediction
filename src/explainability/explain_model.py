"""
Model Explainability Pipeline

This module explains machine learning predictions
using SHAP values and feature importance.

"""

import os
import sys

import pandas as pd

import joblib

import shap

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



# Model path

MODEL_PATH = (
    "models/"
    "water_failure_model.pkl"
)



# Explainability output paths

OUTPUT_PATH = (
    "reports/explainability/"
)


FEATURE_IMPORTANCE_PATH = (
    "reports/explainability/"
    "feature_importance.csv"
)



SHAP_SUMMARY_PATH = (
    "reports/explainability/"
    "shap_summary.png"
)



SHAP_IMPORTANCE_PATH = (
    "reports/explainability/"
    "shap_feature_importance.png"
)



def load_model():

    # Load trained machine learning model.

    try:

        logger.info(
            "Loading trained machine learning model"
        )


        model = joblib.load(
            MODEL_PATH
        )


        logger.info(
            "Model loaded successfully"
        )


        return model


    except Exception as error:


        logger.exception(
            f"Failed loading model: {error}"
        )

        raise



def load_data():

    # Load processed dataset.

    try:

        logger.info(
            "Loading processed dataset for explainability"
        )


        df = pd.read_csv(
            INPUT_PATH
        )


        logger.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )


        return df


    except Exception as error:


        logger.exception(
            f"Failed loading dataset: {error}"
        )

        raise



def prepare_features(df):

    # Select model features.

    try:

        logger.info(
            "Preparing features for explanation"
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
            f"Feature preparation completed. Features: {X.shape[1]}"
        )


        return X


    except Exception as error:


        logger.exception(
            f"Failed preparing features: {error}"
        )

        raise



def generate_feature_importance(model, features):

    # Generate model feature importance.

    try:

        logger.info(
            "Generating feature importance"
        )


        importance = pd.DataFrame(

            {

                "feature": features.columns,

                "importance": model.feature_importances_

            }

        )


        importance = (

            importance
            .sort_values(
                by="importance",
                ascending=False
            )

        )


        os.makedirs(
            OUTPUT_PATH,
            exist_ok=True
        )


        importance.to_csv(
            FEATURE_IMPORTANCE_PATH,
            index=False
        )


        logger.info(
            "Feature importance saved successfully"
        )


        return importance


    except Exception as error:


        logger.exception(
            f"Failed generating feature importance: {error}"
        )

        raise



def generate_shap_values(model, X):

    # Generate SHAP explanations.

    try:

        logger.info(
            "Generating SHAP explanations"
        )


        sample = (

            X.sample(
                1000,
                random_state=42
            )

        )


        explainer = shap.TreeExplainer(

            model,

            feature_perturbation="tree_path_dependent"

        )


        shap_values = explainer.shap_values(
            sample
        )


        logger.info(
            "SHAP values generated successfully"
        )


        return shap_values, sample


    except Exception as error:


        logger.exception(
            f"Failed generating SHAP values: {error}"
        )

        raise



def save_shap_plots(shap_values, sample):

    # Save SHAP visualisations.

    try:

        logger.info(
            "Creating SHAP visualisations"
        )


        os.makedirs(
            OUTPUT_PATH,
            exist_ok=True
        )


        plt.figure(
            figsize=(10, 6)
        )


        shap.summary_plot(

            shap_values,

            sample,

            show=False

        )


        plt.savefig(

            SHAP_SUMMARY_PATH,

            bbox_inches="tight"

        )


        plt.close()



        plt.figure(
            figsize=(10, 6)
        )


        shap.summary_plot(

            shap_values,

            sample,

            plot_type="bar",

            show=False

        )


        plt.savefig(

            SHAP_IMPORTANCE_PATH,

            bbox_inches="tight"

        )


        plt.close()



        logger.info(
            "SHAP visualisations saved successfully"
        )


    except Exception as error:


        logger.exception(
            f"Failed saving SHAP plots: {error}"
        )

        raise



def main():

    # Execute model explainability pipeline.

    try:

        logger.info(
            "Starting model explainability pipeline"
        )


        model = load_model()


        df = load_data()


        X = prepare_features(
            df
        )


        generate_feature_importance(

            model,

            X

        )


        shap_values, sample = generate_shap_values(

            model,

            X

        )


        save_shap_plots(

            shap_values,

            sample

        )


        logger.info(
            "Model explainability pipeline completed successfully"
        )


    except Exception as error:


        logger.exception(
            f"Explainability pipeline failed: {error}"
        )

        raise



if __name__ == "__main__":

    main()