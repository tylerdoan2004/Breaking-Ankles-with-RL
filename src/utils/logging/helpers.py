"""
This module provides helper functions for logging experiment data.
"""
import importlib.metadata
import platform
import shutil
import sys
import numpy as np
import torch
from pathlib import Path
from typing import cast, Callable, Iterable, Literal, Optional, Union
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.grid_dimensions import GridDimensions
from src.utils.environment.helpers import chebyshev_distance
from src.utils.logging.experiment_metadata import ExperimentMetadata, HardwareMetadata, PackageMetadata, PythonMetadata, RuntimeMetadata, SoftwareMetadata
from src.utils.typing.agent import ActType, Agent, ObsType
from src.utils.typing.environment import VideoRecordableEnvironmentFactory


INFO_KEYWORDS: tuple[str, ...] = (
    "outcome",
    "net_progress",
    "path_efficiency",
    "minimum_distance_to_obstacle",
    "minimum_distance_to_boundary",
    "minimum_distance_to_seeker",
    "collision_type"
)


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
        "metadata/configs/validation",
        "metadata/configs/evaluation",
        "models",
        "models/checkpoints",
        "training",
        "training/metrics",
        "training/metrics/tensorboard",
        "validation",
        "validation/metrics",
        "evaluation",
        "evaluation/in_distribution",
        "evaluation/in_distribution/metrics",
        "evaluation/out_of_distribution",
        "evaluation/out_of_distribution/metrics",
    }

    experiment_directory = Path(logging_directory) / experiment_metadata.experiment_name / experiment_metadata.model.name / experiment_metadata.timestamp.strftime("%Y-%m-%d-%H-%M-%S-%f")
    experiment_directory.mkdir(parents = True, exist_ok = True)

    for subdirectory in SUBDIRECTORIES:
        (experiment_directory / subdirectory).mkdir(parents = True, exist_ok = True)

    # Copy experiment configs to experiment directory
    shutil.copyfile(experiment_metadata.system_configurations.training.path, experiment_directory / "metadata/configs/training/training.yaml")
    shutil.copyfile(experiment_metadata.system_configurations.validation.path, experiment_directory / "metadata/configs/validation/validation.yaml")
    shutil.copyfile(experiment_metadata.system_configurations.evaluation.in_distribution.path, experiment_directory / "metadata/configs/evaluation/in_distribution.yaml")
    shutil.copyfile(experiment_metadata.system_configurations.evaluation.out_of_distribution.path, experiment_directory / "metadata/configs/evaluation/out_of_distribution.yaml")

    # Write experiment metadata to experiment directory
    experiment_metadata.to_yaml(experiment_directory / "metadata/experiment_metadata.yaml")
    return experiment_directory


def record_video_single_episode(*,
                                video_directory: str, video_name_prefix: str,
                                environment_factory: VideoRecordableEnvironmentFactory[Env[ObsType, ActType]], seed: int,
                                agent: Optional[Agent[ObsType, ActType]] = None, deterministic: bool = True) -> None:
    """
    Records a video of a single episode of an agent performing actions in an environment.
    
    :param video_directory: The directory used to store the video.
    :param video_name_prefix: The prefix used to name the video.
    :param environment_factory: A function that creates a video recordable environment.
    :param seed: The seed to use for the environment.
    :param agent: The agent to record the video for.
    :param deterministic: Whether the agent predicts actions deterministically.
    :return: None.
    """
    environment: Env[ObsType, ActType] = environment_factory(render_mode = "rgb_array")
    try:
        environment = RecordVideo(
            env = environment,
            video_folder = video_directory,
            episode_trigger = lambda episode: episode == 0,
            name_prefix = video_name_prefix
        )
        observation, _ = environment.reset(seed = seed)
        done = False
        while not done:
            action: object
            if agent is None:
                action = environment.action_space.sample()
            else:
                action = agent.predict(observation, deterministic = deterministic)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()
            observation, _, terminated, truncated, _ = environment.step(cast(ActType, action))
            done = terminated or truncated
    finally:
        environment.close()


def compute_distance_to_hazards(*,
                                current_agent_coordinates: Coordinates,
                                obstacles_coordinates: Iterable[Coordinates],
                                grid_dimensions: GridDimensions,
                                seekers_coordinates: Iterable[Coordinates],
                                distance_metric: Callable[[Coordinates, Coordinates], int | float] = chebyshev_distance) -> dict[Literal["obstacle", "boundary", "seeker"], int | float]:
    """
    Computes the agent's distance to each type of hazard.
    
    :param current_agent_coordinates: The current coordinates of the agent.
    :param obstacles_coordinates: The coordinates of the obstacles.
    :param grid_dimensions: The grid dimensions.
    :param seekers_coordinates: The coordinates of the seekers.
    :return: A dictionary containing the agent's distance to each type of hazard.
    """
    return {
        "obstacle": min(distance_metric(current_agent_coordinates, obstacle_coordinates) for obstacle_coordinates in obstacles_coordinates),
        "boundary": min(current_agent_coordinates.x + 1, grid_dimensions.width - current_agent_coordinates.x, current_agent_coordinates.y + 1, grid_dimensions.height - current_agent_coordinates.y),
        "seeker": min(distance_metric(current_agent_coordinates, seeker_coordinates) for seeker_coordinates in seekers_coordinates),
    }


def compute_minimum_distance_to_hazards(*,
                                        current_minimum_distance_to_hazards: dict[Literal["obstacle", "boundary", "seeker"], int | float],
                                        current_distance_to_hazards: dict[Literal["obstacle", "boundary", "seeker"], int | float]) -> dict[Literal["obstacle", "boundary", "seeker"], int | float]:
    """
    Computes the agent's minimum distance to each type of hazard.
    
    :param current_minimum_distance_to_hazards: The current minimum distance to each type of hazard.
    :param current_distance_to_hazards: The current distance to each type of hazard.
    :return: A dictionary containing the agent's minimum distance to each type of hazard.
    """
    return {
        key: min(current_minimum_distance_to_hazards[key], value)
        for key, value in current_distance_to_hazards.items()
    }


def compute_net_progress(starting_time_steps_to_goal: Optional[int], ending_time_steps_to_goal: Optional[int]) -> Optional[int]:
    """
    Computes the net progress of the agent over an episode.
    
    :param starting_time_steps_to_goal: The minimum number of time steps to the goal at the start of the episode.
    :param ending_time_steps_to_goal: The minimum number of time steps to the goal at the end of the episode.
    :return: The net progress of the agent over the episode.
    """
    if starting_time_steps_to_goal is None or ending_time_steps_to_goal is None:
        return None
    return starting_time_steps_to_goal - ending_time_steps_to_goal


def compute_path_efficiency(outcome: Optional[str], starting_time_steps_to_goal: Optional[int], episode_length: int) -> Optional[float]:
    """
    Computes the path efficiency of the agent over an episode. The path efficiency is the ratio of the minimum number of time steps to the goal 
    at the start of the episode to the time step at which the episode ended. The path efficiency is only valid if the agent reached the goal.
    
    :param outcome: The outcome of the episode.
    :param starting_time_steps_to_goal: The minimum number of time steps to the goal at the start of the episode.
    :param episode_length: The length of the episode.
    :return: The path efficiency of the agent over the episode.
    """
    if outcome != "goal" or starting_time_steps_to_goal is None:
        return None
    return starting_time_steps_to_goal / episode_length


def compute_episode_metrics(*,
                            outcome: Optional[str],
                            starting_time_steps_to_goal: Optional[int],
                            ending_time_steps_to_goal: Optional[int],
                            episode_length: int,
                            minimum_distance_to_hazards: dict[Literal["obstacle", "boundary", "seeker"], int | float],
                            collision_type: Optional[Literal["obstacle", "boundary", "seeker"]]) -> dict[str, Optional[Union[int, float, str]]]:
    """
    Computes the metrics for an episode.
    
    :param outcome: The outcome of the episode.
    :param starting_time_steps_to_goal: The minimum number of time steps to the goal at the start of the episode.
    :param ending_time_steps_to_goal: The minimum number of time steps to the goal at the end of the episode.
    :param episode_length: The length of the episode.
    :param minimum_distance_to_hazards: A dictionary mapping the type of hazard to the agent's minimum distance to the hazard.
    :param collision_type: The type of collision that occurred during the episode.
    :return: A dictionary of episode metrics.
    """
    episode_metrics: dict[str, Optional[Union[int, float, str]]] = {}
    episode_metrics["outcome"] = outcome
    episode_metrics["net_progress"] = compute_net_progress(starting_time_steps_to_goal, ending_time_steps_to_goal)
    episode_metrics["path_efficiency"] = compute_path_efficiency(outcome, starting_time_steps_to_goal, episode_length)
    episode_metrics["minimum_distance_to_obstacle"] = minimum_distance_to_hazards.get("obstacle")
    episode_metrics["minimum_distance_to_boundary"] = minimum_distance_to_hazards.get("boundary")
    episode_metrics["minimum_distance_to_seeker"] = minimum_distance_to_hazards.get("seeker")
    episode_metrics["collision_type"] = collision_type
    return episode_metrics
