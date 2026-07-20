"""
Test suite for model explainability pipeline.

This module validates that explainability
artifacts are generated correctly.
"""


import os
import pandas as pd



FEATURE_IMPORTANCE_PATH = (
    "reports/"
    "explainability/"
    "feature_importance.csv"
)


SHAP_SUMMARY_PATH = (
    "reports/"
    "explainability/"
    "shap_summary.png"
)


SHAP_FEATURE_PATH = (
    "reports/"
    "explainability/"
    "shap_feature_importance.png"
)



def test_feature_importance_exists():

    """
    Validate feature importance output.
    """

    assert os.path.exists(
        FEATURE_IMPORTANCE_PATH
    )



def test_feature_importance_schema():

    """
    Validate feature importance structure.
    """

    df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )


    assert "feature" in df.columns

    assert "importance" in df.columns



def test_feature_importance_values():

    """
    Validate importance values.
    """

    df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )


    assert (
        df["importance"]
        .between(0, 1)
        .all()
    )



def test_shap_visualisations_exist():

    """
    Validate SHAP explanation visualisations.
    """

    assert os.path.exists(
        SHAP_SUMMARY_PATH
    )

    assert os.path.exists(
        SHAP_FEATURE_PATH
    )



def test_explainability_figures_exist():

    """
    Validate explainability directory contains outputs.
    """

    files = os.listdir(
        "reports/explainability"
    )


    assert len(files) >= 3