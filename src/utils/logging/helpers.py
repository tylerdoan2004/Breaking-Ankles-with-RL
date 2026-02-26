import importlib.metadata
import platform
import shutil
import sys
import torch
from pathlib import Path
from typing import cast, Optional
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from src.utils.logging.experiment_metadata import ExperimentMetadata, HardwareMetadata, PackageMetadata, PythonMetadata, RuntimeMetadata, SoftwareMetadata
from src.utils.typing.agent import ActType, Agent, ObsType
from src.utils.typing.environment import VideoRecordableEnvironmentFactory


def get_package_metadata(package_name: str) -> PackageMetadata:
    """
    Returns metadata about a package.
    
    :param package_name: The name of the package to get metadata about.
    :return: Metadata about a package.
    """
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return PackageMetadata(version = version)


def get_runtime_metadata() -> RuntimeMetadata:
    """
    Returns metadata about the runtime environment of the experiment.
    
    :return: Metadata about the runtime environment of the experiment.
    """
    hardware = HardwareMetadata(
        platform = platform.platform(),
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )
    software = SoftwareMetadata(
        python = PythonMetadata(
            implementation = platform.python_implementation(),
            version = sys.version
        ),
        gymnasium = get_package_metadata("gymnasium"),
        minigrid = get_package_metadata("minigrid"),
        numpy = get_package_metadata("numpy"),
        pyyaml = get_package_metadata("PyYAML"),
        stable_baselines3 = get_package_metadata("stable-baselines3")
    )
    return RuntimeMetadata(
        hardware = hardware,
        software = software
    )


def initialize_logging_directories(*, logging_directory: str, experiment_metadata: ExperimentMetadata) -> Path:
    """
    Sets up the logging directories for the experiment.
    
    :param logging_directory: The directory used to store experiment logs.
    :param experiment_metadata: The metadata for the experiment.
    :return: The path to the logging directory for the experiment.
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
        "evaluation/out_of_distribution",
        "evaluation/out_of_distribution/metrics",
        "evaluation/out_of_distribution/videos"
    }

    experiment_directory = Path(logging_directory) / experiment_metadata.experiment_name / experiment_metadata.model.name / experiment_metadata.timestamp.strftime("%Y-%m-%d-%H-%M-%S-%f")
    experiment_directory.mkdir(parents = True, exist_ok = True)

    for subdirectory in SUBDIRECTORIES:
        (experiment_directory / subdirectory).mkdir(parents = True, exist_ok = True)

    # Copy experiment configs to experiment directory
    shutil.copyfile(experiment_metadata.system_configurations.training.path, experiment_directory / "metadata/configs/training/training.yaml")
    shutil.copyfile(experiment_metadata.system_configurations.evaluation.in_distribution.path, experiment_directory / "metadata/configs/evaluation/in_distribution.yaml")
    shutil.copyfile(experiment_metadata.system_configurations.evaluation.out_of_distribution.path, experiment_directory / "metadata/configs/evaluation/out_of_distribution.yaml")

    # Write experiment metadata to experiment directory
    experiment_metadata.to_yaml(experiment_directory / "metadata/experiment_metadata.yaml")
    return experiment_directory


def record_video_single_episode(*,
                                video_directory: str, video_name_prefix: str,
                                environment_factory: VideoRecordableEnvironmentFactory[Env[ObsType, ActType]],
                                agent: Optional[Agent[ObsType, ActType]] = None, deterministic: bool = True) -> None:
    """
    Records a video of a single episode of an agent performing actions in an environment.
    
    :param video_directory: The directory used to store the video.
    :param video_name_prefix: The prefix used to name the video.
    :param environment_factory: A function that creates a video recordable environment.
    :param agent: The agent to record the video for.
    :param deterministic: Whether the agent predicts actions deterministically.
    :return: None.
    """
    environment: Env[ObsType, ActType] = environment_factory(render_mode = "rgb_array")
    environment = RecordVideo(
        env = environment,
        video_folder = video_directory,
        episode_trigger = lambda episode: episode == 0,
        name_prefix = video_name_prefix
    )
    environment = cast(
        Env[ObsType, ActType],
        environment
    )

    observation, _ = environment.reset()
    done = False
    while not done:
        if agent is None:
            action = environment.action_space.sample()
        else:
            action = agent.predict(observation, deterministic = deterministic)
        observation, _, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated

    environment.close()
