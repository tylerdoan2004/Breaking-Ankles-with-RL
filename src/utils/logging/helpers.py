import importlib.metadata
import platform
import shutil
import sys
import torch
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, TypeVar
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from src.utils.logging.experiment_metadata import ExperimentMetadata, HardwareMetadata, PackageMetadata, PythonMetadata, RuntimeMetadata, SoftwareMetadata


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


EnvTypeCovariant = TypeVar("EnvTypeCovariant", bound = Env, covariant = True)
class VideoRecordableEnvironmentFactory(Protocol[EnvTypeCovariant]):
    """
    A protocol for creating video recordable environments.
    """
    def __call__(self, *args: Any, render_mode: Literal["rgb_array"], **kwargs: Any) -> EnvTypeCovariant:
        """
        Creates a video recordable environment.
        
        :param render_mode: The render mode to use for the environment.
        :return: A video recordable environment.
        """
        ...


ObsTypeContravariant = TypeVar("ObsTypeContravariant", contravariant = True)
ActTypeCovariant = TypeVar("ActTypeCovariant", covariant = True)
class Agent(Protocol[ObsTypeContravariant, ActTypeCovariant]):
    """
    A protocol for representing an agent.
    """
    def predict(self, observation: ObsTypeContravariant, *, deterministic: bool = True) -> ActTypeCovariant:
        """
        Predicts an action given an observation.
        
        :param observation: The observation used to predict the next action.
        :param deterministic: Whether the agent predicts actions deterministically.
        :return: The predicted action.
        """
        ...


O = TypeVar("O")
A = TypeVar("A")
def record_video_single_episode(*,
                                video_directory: str, video_name_prefix: str,
                                environment_factory: VideoRecordableEnvironmentFactory[Env[O, A]],
                                agent: Optional[Agent[O, A]] = None, deterministic: bool = True) -> None:
    """
    Records a video of a single episode of an agent performing actions in an environment.
    
    :param video_directory: The directory used to store the video.
    :param video_name_prefix: The prefix used to name the video.
    :param environment_factory: A function that creates a video recordable environment.
    :param agent: The agent to record the video for.
    :param deterministic: Whether the agent predicts actions deterministically.
    :return: None.
    """
    environment = environment_factory(render_mode = "rgb_array")
    environment = RecordVideo(
        env = environment,
        video_folder = video_directory,
        episode_trigger = lambda episode: episode == 0,
        name_prefix = video_name_prefix
    )

    observation, _ = environment.reset()
    done = False
    while not done:
        action = agent.predict(observation, deterministic = deterministic) if agent is not None else environment.action_space.sample()
        observation, _, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated

    environment.close()
