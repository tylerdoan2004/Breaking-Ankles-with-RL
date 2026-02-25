import importlib
import importlib.metadata
import platform
import shutil
import sys
from pathlib import Path
from typing import Optional
from src.utils.logging.experiment_metadata import ExperimentMetadata


def get_package_version(package_name: str) -> Optional[str]:
    """
    Returns the installed version of a package, or None if the package is not installed.
    
    :param package_name: The name of the package to get the version of.
    :return: The installed version of a package, or None if the package is not installed.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_runtime_environment_information() -> dict[str, Optional[str]]:
    """
    Returns information about the runtime environment relevant to the experiment.
    
    :return: A dictionary containing information about the runtime environment relevant to the experiment.
    """
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "gymnasium": get_package_version("gymnasium"),
        "minigrid": get_package_version("minigrid"),
        "numpy": get_package_version("numpy"),
        "pyyaml": get_package_version("pyyaml"),
        "stable_baselines3": get_package_version("stable_baselines3"),
    }


def initialize_logging_directories(*, logging_directory: str, experiment_metadata: ExperimentMetadata) -> None:
    """
    Sets up the logging directories for the experiment.
    
    :param logging_directory: The directory used to store experiment logs.
    :param experiment_metadata: The metadata for the experiment.
    :return: None.
    """
    SUBDIRECTORIES = {
        "metadata",
        "metadata/configs",
        "metadata/configs/training",
        "metadata/configs/evaluation",
        "models",
        "models/checkpoints",
        "training",
        "training/metrics",
        "training/metrics/tensorboard",
        "training/videos",
        "training/videos/pretraining",
        "training/videos/training",
        "training/videos/posttraining",
        "evaluation",
        "evaluation/in_distribution",
        "evaluation/in_distribution/metrics",
        "evaluation/in_distribution/videos",
        "evaluation/in_distribution/videos/pretraining",
        "evaluation/in_distribution/videos/training",
        "evaluation/in_distribution/videos/posttraining",
        "evaluation/out_of_distribution",
        "evaluation/out_of_distribution/metrics",
        "evaluation/out_of_distribution/videos",
        "evaluation/out_of_distribution/videos/pretraining",
        "evaluation/out_of_distribution/videos/training",
        "evaluation/out_of_distribution/videos/posttraining",
    }

    experiment_directory = Path(logging_directory) / experiment_metadata.experiment_name / experiment_metadata.model_name / experiment_metadata.timestamp.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_directory.mkdir(parents = True, exist_ok = True)

    for subdirectory in SUBDIRECTORIES:
        (experiment_directory / subdirectory).mkdir(parents = True, exist_ok = True)

    # Copy experiment configs to experiment directory
    shutil.copyfile(experiment_metadata.config_paths["training"], experiment_directory / "metadata/configs/training/training.yaml")
    shutil.copyfile(experiment_metadata.config_paths["evaluation/in_distribution"], experiment_directory / "metadata/configs/evaluation/in_distribution.yaml")
    shutil.copyfile(experiment_metadata.config_paths["evaluation/out_of_distribution"], experiment_directory / "metadata/configs/evaluation/out_of_distribution.yaml")

    # Write experiment metadata to experiment directory
    experiment_metadata.to_yaml(experiment_directory / "metadata/experiment_metadata.yaml")
