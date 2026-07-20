"""
Test suite for the central logging system.

This module validates that the project logging utility
creates valid logger instances for pipeline components.
"""

import os
import sys
import logging


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


from src.utils.logger import get_logger



def test_logger_creation():

    """
    Validate logger creation.
    """

    logger = get_logger(
        "test_logger"
    )


    assert logger is not None


    assert isinstance(
        logger,
        logging.Logger
    )



def test_logger_name():

    """
    Validate logger naming.
    """

    logger = get_logger(
        "pipeline_test"
    )


    assert (
        logger.name
        ==
        "pipeline_test"
    )