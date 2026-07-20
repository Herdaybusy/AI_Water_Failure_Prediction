"""
Test suite for raw data quality profiling pipeline.

This module validates the data quality checks
performed before ETL processing.
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


from src.data_quality.profile_raw_data import (
    check_missing_values,
    check_duplicates,
    check_sensor_ranges
)



def create_test_dataset():

    """
    Create sample telemetry data
    for testing data quality functions.
    """

    data = {

        "pressure_bar": [
            3.5,
            4.0,
            -1.0
        ],

        "flow_rate_lps": [
            120,
            130,
            140
        ],

        "vibration_mm_s": [
            2.5,
            3.0,
            4.0
        ],

        "turbidity_ntu": [
            1.2,
            None,
            2.1
        ]

    }


    return pd.DataFrame(data)



def test_missing_value_detection():

    """
    Validate missing value detection.
    """

    df = create_test_dataset()


    missing = check_missing_values(
        df
    )


    assert (
        missing["turbidity_ntu"]
        ==
        1
    )



def test_duplicate_detection():

    """
    Validate duplicate record detection.
    """

    df = create_test_dataset()


    duplicates = check_duplicates(
        df
    )


    assert duplicates == 0



def test_sensor_validation():

    """
    Validate invalid sensor readings.
    """

    df = create_test_dataset()


    anomalies = check_sensor_ranges(
        df
    )


    assert (
        anomalies["pressure_bar_invalid"]
        ==
        1
    )