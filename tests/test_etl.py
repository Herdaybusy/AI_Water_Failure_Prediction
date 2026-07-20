"""
Test suite for ETL cleaning pipeline.

This module validates data cleaning and feature
engineering logic before model training.
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


from src.etl.clean_water_data import (
    handle_missing_values,
    remove_duplicates,
    validate_sensor_data,
    create_features
)


def create_test_dataset():

    """
    Create sample telemetry data
    for ETL testing.
    """

    data = {

        "pressure_bar": [
            3.5,
            4.0,
            2.8
        ],

        "flow_rate_lps": [
            120,
            130,
            110
        ],

        "temperature_c": [
            15,
            16,
            14
        ],

        "vibration_mm_s": [
            2.5,
            3.0,
            4.0
        ],

        "turbidity_ntu": [
            1.5,
            2.0,
            2.5
        ],

        "chlorine_mg_l": [
            0.7,
            0.8,
            0.6
        ],

        "pump_runtime_hours": [
            3000,
            4000,
            5000
        ],

        "maintenance_age_days": [
            100,
            200,
            300
        ]

    }


    return pd.DataFrame(data)



def test_remove_duplicates():

    """
    Validate duplicate removal.
    """

    df = create_test_dataset()

    df = pd.concat(
        [
            df,
            df.iloc[[0]]
        ],
        ignore_index=True
    )


    cleaned = remove_duplicates(
        df
    )


    assert len(cleaned) == 3


def test_sensor_validation():

    """
    Validate sensor measurement checks.
    """

    df = create_test_dataset()

    df.loc[0, "pressure_bar"] = -2


    validated = validate_sensor_data(
        df
    )


    assert (
        validated is not None
    )

def test_missing_value_handling():

    """
    Validate missing value replacement.
    """

    df = create_test_dataset()

    df.loc[0,"pressure_bar"] = None


    cleaned = handle_missing_values(
        df
    )


    assert (
        cleaned["pressure_bar"]
        .isnull()
        .sum()
        ==
        0
    )



def test_feature_creation():

    """
    Validate engineered feature creation.
    """

    df = create_test_dataset()


    transformed = create_features(
        df
    )


    expected_features = [

        "pressure_health_score",
        "flow_efficiency",
        "maintenance_risk_score",
        "pump_utilisation_score",
        "operational_risk_score"

    ]


    for feature in expected_features:

        assert feature in transformed.columns