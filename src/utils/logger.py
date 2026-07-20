"""
Central logging configuration for the Water Network Failure Prediction project.

All pipeline components use this logger instead of print statements.

Logs are written to:
1. Console (during development)
2. Log files (for tracking production runs)

Example log:
2026-07-18 10:30:15 | INFO | etl.clean_water_data | Loading raw dataset
"""

import logging
import os
from datetime import datetime


# Create logs folder automatically if it does not exist
LOG_FOLDER = "logs"

os.makedirs(
    LOG_FOLDER,
    exist_ok=True
)


# Each execution gets its own dated log file
LOG_FILE = os.path.join(
    LOG_FOLDER,
    f"water_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
)



def get_logger(name):
    """
    Creates and returns a reusable logger.

    Parameters:
        name:
            Usually __name__ from the calling module.

    Returns:
        Configured logger instance
    """


    logger = logging.getLogger(name)


    # Avoid duplicate messages
    # when modules are imported multiple times
    if logger.handlers:
        return logger


    logger.setLevel(
        logging.INFO
    )


    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )


    # Display logs in terminal
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )


    # Save logs permanently
    file_handler = logging.FileHandler(
        LOG_FILE
    )

    file_handler.setFormatter(
        formatter
    )


    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )


    return logger