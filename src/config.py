import os
import yaml


def load_config(config_path="./config/config.yaml"):
    """
    Loads the YAML configuration file.

    This function loads the configuration file in YAML format, parses its content, and returns
    the configuration as a dictionary. If the file is missing or cannot be parsed correctly,
    appropriate exceptions are raised.

    Parameters:
    -----------
    config_path : str, optional
        The path to the configuration file. Defaults to './config/config.yaml'.

    Returns:
    --------
    dict
        A dictionary containing the parsed configuration values.

    Raises:
    -------
    FileNotFoundError:
        If the configuration file is not found at the specified path.

    ValueError:
        If there is an error while parsing the YAML configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML configuration: {exc}")

    return config
