"""
Test suite for machine learning model training pipeline.

This module validates that the trained model artifact
and evaluation outputs are generated correctly.
"""


import os
import sys
import joblib


# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)



MODEL_PATH = (
    "models/"
    "water_failure_model.pkl"
)


METRICS_PATH = (
    "reports/metrics/"
    "model_metrics.txt"
)



def test_model_artifact_exists():

    """
    Validate trained model file exists.
    """

    assert os.path.exists(
        MODEL_PATH
    )



def test_model_can_be_loaded():

    """
    Validate saved model can be loaded.
    """

    model = joblib.load(
        MODEL_PATH
    )


    assert model is not None



def test_model_metrics_exist():

    """
    Validate model evaluation metrics were generated.
    """

    assert os.path.exists(
        METRICS_PATH
    )