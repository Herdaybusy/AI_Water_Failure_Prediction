"""
Configuration loader.

Loads project settings from YAML file.
"""


import yaml


def load_config():

    """
    Read configuration settings.

    Returns:
        dict: Project configuration
    """

    with open(
        "config/config.yaml",
        "r"
    ) as file:

        config = yaml.safe_load(file)

    return config