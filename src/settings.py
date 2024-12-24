import os
from dataclasses import dataclass

import toml

__all__ = [
    "PathSettings",
    "EnvironmentSettings",
    "load_environment_settings",
]


@dataclass
class PathSettings:
    """
    Path settings on a specific platform.
    """

    name: str
    train: str
    test: str
    sample: str


@dataclass
class EnvironmentSettings:
    """
    Collection of environment settings, including paths and hyperparameters.
    """

    paths: PathSettings


def load_environment_settings(path: str | os.PathLike) -> EnvironmentSettings:
    """
    Load environment settings from a configuration file.

    Args:
        path: the path to the configuration file.

    Returns:
        an EnvironmentSettings object if loaded successfully
    """
    with open(path, encoding="utf-8") as file:
        settings = toml.load(file)

    # load enabled path configuration
    paths: PathSettings | None = None
    for configuration in settings["path"]["configurations"]:
        if configuration["name"] == settings["path"]["enabled"]:
            paths = PathSettings(**configuration)
            break
    assert paths is not None, "invalid path configuration"

    return EnvironmentSettings(paths)
