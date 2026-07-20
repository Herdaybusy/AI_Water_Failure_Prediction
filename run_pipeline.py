"""
AI Water Network Failure Prediction Pipeline Orchestrator

This module executes the complete end-to-end workflow,
from raw data generation through model prediction.

Each pipeline stage is maintained as an independent module,
while this script manages execution order and monitoring.

"""

import os
import sys
import subprocess

from src.utils.logger import get_logger


# Ensure project modules are available when running from root
PROJECT_ROOT = os.path.abspath(
    os.path.dirname(__file__)
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


logger = get_logger(__name__)


PIPELINE_STAGES = [

    {
        "name": "Generate water telemetry data",
        "script": "src/data_generation/generate_raw_data.py"
    },

    {
        "name": "Profile raw dataset quality",
        "script": "src/data_quality/profile_raw_data.py"
    },

    {
        "name": "Clean data and engineer features",
        "script": "src/etl/clean_water_data.py"
    },

    {
        "name": "Run exploratory data analysis",
        "script": "src/ml/explore_training_data.py"
    },

    {
        "name": "Train failure prediction model",
        "script": "src/ml/train_failure_model.py"
    },

    {
        "name": "Generate model explainability outputs",
        "script": "src/explainability/explain_model.py"
    },

    {
        "name": "Generate failure predictions",
        "script": "src/ml/predict_failure.py"
    }

]


def run_stage(stage):
    """
    Execute an individual pipeline stage.
    """

    stage_name = stage["name"]
    script = stage["script"]

    logger.info(
        f"Running stage: {stage_name}"
    )

    try:

        if not os.path.exists(script):

            raise FileNotFoundError(
                f"Pipeline script not found: {script}"
            )
        
        subprocess.run(

            [
                sys.executable,
                script
            ],

            check=True

        )

        logger.info(
            f"Completed stage: {stage_name}"
        )


    except subprocess.CalledProcessError:

        logger.exception(
            f"Stage failed: {stage_name}"
        )

        raise



def run_pipeline():
    """
    Execute all pipeline stages in sequence.
    """

    logger.info(
        "Starting AI water network failure prediction pipeline"
    )


    for stage in PIPELINE_STAGES:

        run_stage(stage)


    logger.info(
        "Pipeline execution completed successfully"
    )



def main():

    try:

        run_pipeline()


    except Exception as error:

        logger.exception(
            f"Pipeline failed: {error}"
        )

        sys.exit(1)



if __name__ == "__main__":

    main()