"""
A module for creating reactive avoidance environments.
"""
from pathlib import Path
from typing import Literal, Optional
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from src.environment import ReactiveAvoidanceEnv
from src.utils.typing.environment import VideoRecordableEnvironmentFactory
from src.utils.yaml_parser.configuration import SystemConfiguration


def make_env_from_yaml(yaml_file_path: str, *, render_mode: Optional[str] = None) -> ReactiveAvoidanceEnv:
    """
    Makes a reactive avoidance environment from a YAML file.

    :param yaml_file_path: The path to the YAML file.
    :param render_mode: The rendering mode for the environment.
    :return: The reactive avoidance environment.
    """
    config = SystemConfiguration.parse_config_file(Path(yaml_file_path))
    environment = ReactiveAvoidanceEnv(config, render_mode)
    return environment


def validate_environment(yaml_file_path: str) -> None:
    """
    Validates a reactive avoidance environment from a YAML file.

    :param yaml_file_path: The path to the YAML file.
    :return: None.
    """
    environment = make_env_from_yaml(yaml_file_path)
    check_env(environment, warn = True)
    environment.close()


def make_vectorized_environment(*, yaml_file_path: str, n_envs: int, seed: int) -> VecEnv:
    """
    Creates a vectorized environment without auto-monitor wrapping around each environment.

    :param yaml_file_path: The path to the YAML file defining the environment.
    :param n_envs: The number of environments to create.
    :param seed: The seed to use for the environment.
    :return: A vectorized environment.
    """
    environment_factories = [lambda: make_env_from_yaml(yaml_file_path) for _ in range(n_envs)]
    vectorized_environment_class = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vectorized_environment = vectorized_environment_class(list(environment_factories))
    vectorized_environment.seed(seed)
    return vectorized_environment


def make_video_recordable_environment_factory(yaml_file_path: str) -> VideoRecordableEnvironmentFactory[ReactiveAvoidanceEnv]:
    """
    Creates a factory for creating video recordable environments.

    :param yaml_file_path: The path to the YAML file defining the environment.
    :return: A factory for creating video recordable environments from the YAML file.
    """
    def video_recordable_environment_factory(*, render_mode: Literal["rgb_array"]) -> ReactiveAvoidanceEnv:
        """
        Creates a video recordable environment.

        :param render_mode: The render mode to use for the environment. Must be "rgb_array".
        :return: A video recordable environment.
        """
        return make_env_from_yaml(yaml_file_path, render_mode = render_mode)
    return video_recordable_environment_factory
