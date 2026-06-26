from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        A dictionary containing configuration items.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)
