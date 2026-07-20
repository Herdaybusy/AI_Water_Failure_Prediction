"""
Test suite for failure prediction pipeline.

This module validates prediction outputs
generated from the trained model.
"""


import os
import sys
import pandas as pd


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



PREDICTION_PATH = (
    "results/"
    "predictions.csv"
)



def test_prediction_file_exists():

    """
    Validate prediction output exists.
    """

    assert os.path.exists(
        PREDICTION_PATH
    )



def test_prediction_output_columns():

    """
    Validate prediction output schema.
    """

    df = pd.read_csv(
        PREDICTION_PATH
    )


    expected_columns = [

        "asset_id",
        "failure_probability",
        "risk_level",
        "recommendation"

    ]


    for column in expected_columns:

        assert column in df.columns



def test_prediction_probability_range():

    """
    Validate probability values.
    """

    df = pd.read_csv(
        PREDICTION_PATH
    )


    assert (
        df["failure_probability"]
        .between(0, 1)
        .all()
    )



def test_prediction_risk_levels():

    """
    Validate generated risk categories.
    """

    df = pd.read_csv(
        PREDICTION_PATH
    )


    valid_levels = [

        "LOW",
        "MEDIUM",
        "HIGH"

    ]


    assert (
        df["risk_level"]
        .isin(valid_levels)
        .all()
    )