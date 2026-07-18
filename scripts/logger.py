"""
Logging configuration module.

This module provides a central logging system
used across the entire AI pipeline.
"""


from pathlib import Path
from loguru import logger
import sys
import yaml


def setup_logger():
    """
    Configure application logging.

    Logs are written to both:
    - Console
    - Log file
    """

    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)


    log_folder = Path(
        config["logging"]["folder"]
    )

    log_folder.mkdir(
        exist_ok=True
    )


    logger.remove()


    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{message}"
        )
    )


    logger.add(
        log_folder / "pipeline.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{message}"
        )
    )


    return logger