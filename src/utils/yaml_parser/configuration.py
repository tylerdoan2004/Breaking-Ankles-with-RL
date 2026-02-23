"""
This module provides the SystemConfiguration, AgentConfiguration, SeekerConfiguration, and EnvironmentConfiguration classes for representing system, agent, seeker, and environment configurations.
"""
from dataclasses import dataclass
from pathlib import Path
from utils.yaml_parser.helpers import (
    is_agent_config_shallowly_valid,
    is_coordinates_data_valid,
    is_environment_config_shallowly_valid,
    is_grid_dimensions_data_valid,
    is_obstacles_coordinates_data_valid,
    is_seekers_config_shallowly_valid,
    is_system_config_shallowly_valid,
    is_yaml_file,
    load_yaml_file,
    validate_system_configuration
)
from utils.environment.coordinates import Coordinates
from utils.environment.grid_dimensions import GridDimensions


__all__ = ["SystemConfiguration", "AgentConfiguration", "SeekerConfiguration", "EnvironmentConfiguration"]


@dataclass(frozen = True, kw_only = True)
class AgentConfiguration:
    """
    A class for representing the configuration of an agent in a two-dimensional gridworld.
    """
    start_coordinates: Coordinates
    goal_coordinates: Coordinates
    velocity: int
    visibility_radius: int
    observation_stack_depth: int

    @staticmethod
    def from_dict(agent_config: dict) -> "AgentConfiguration":
        """
        Creates an AgentConfiguration object from a shallowly valid agent configuration dictionary.
        
        :param agent_config: A dictionary representing a shallowly valid agent configuration.
        :return: An AgentConfiguration object.
        """
        if not is_coordinates_data_valid(agent_config["start_coordinates"]):
            raise ValueError(f"The agent configuration does not have valid start coordinates: {agent_config}")

        if not is_coordinates_data_valid(agent_config["goal_coordinates"]):
            raise ValueError(f"The agent configuration does not have valid goal coordinates: {agent_config}")

        return AgentConfiguration(
            start_coordinates = Coordinates.from_list(agent_config["start_coordinates"]),
            goal_coordinates = Coordinates.from_list(agent_config["goal_coordinates"]),
            velocity = agent_config["velocity"],
            visibility_radius = agent_config["visibility_radius"],
            observation_stack_depth = agent_config["observation_stack_depth"]
        )


@dataclass(frozen = True, kw_only = True)
class SeekerConfiguration:
    """
    A class for representing the configuration of a seeker in a two-dimensional gridworld.
    """
    start_coordinates: Coordinates
    velocity: int

    @staticmethod
    def from_dict(seeker_config: dict) -> "SeekerConfiguration":
        """
        Creates a SeekerConfiguration object from a shallowly valid seeker configuration dictionary.
        
        :param seeker_config: A dictionary representing a shallowly valid seeker configuration.
        :return: A SeekerConfiguration object.
        """
        if not is_coordinates_data_valid(seeker_config["start_coordinates"]):
            raise ValueError(f"The seeker configuration does not have valid start coordinates: {seeker_config}")

        return SeekerConfiguration(
            start_coordinates = Coordinates.from_list(seeker_config["start_coordinates"]),
            velocity = seeker_config["velocity"]
        )


@dataclass(frozen = True, kw_only = True)
class EnvironmentConfiguration:
    """
    A class for representing the configuration of a two-dimensional gridworld.
    """
    grid_dimensions: GridDimensions
    obstacles_coordinates: list[Coordinates]
    episode_time_limit: int

    @staticmethod
    def from_dict(environment_config: dict) -> "EnvironmentConfiguration":
        """
        Creates an EnvironmentConfiguration object from a shallowly valid environment configuration dictionary.
        
        :param environment_config: A dictionary representing a shallowly valid environment configuration.
        :return: An EnvironmentConfiguration object.
        """
        if not is_grid_dimensions_data_valid(environment_config["grid_dimensions"]):
            raise ValueError(f"The environment configuration does not have valid grid dimensions: {environment_config}")

        if not is_obstacles_coordinates_data_valid(environment_config["obstacles_coordinates"]):
            raise ValueError(f"The environment configuration does not have valid coordinates for obstacles: {environment_config}")

        return EnvironmentConfiguration(
            grid_dimensions = GridDimensions.from_tuple((environment_config["grid_dimensions"]["width"], environment_config["grid_dimensions"]["height"])),
            obstacles_coordinates = [Coordinates.from_list(coordinates) for coordinates in environment_config["obstacles_coordinates"]],
            episode_time_limit = environment_config["episode_time_limit"]
        )


@dataclass(frozen = True, kw_only = True)
class SystemConfiguration:
    """
    A class for representing the configuration of an agent-seeker-gridworld system.
    """
    agent: AgentConfiguration
    seekers: list[SeekerConfiguration]
    environment: EnvironmentConfiguration
    num_seekers: int

    @staticmethod
    def from_dict(system_config: dict) -> "SystemConfiguration":
        """
        Creates a SystemConfiguration object from a shallowly valid system configuration dictionary.
        
        :param system_config: A dictionary representing a shallowly valid system configuration.
        :return: A SystemConfiguration object.
        """
        if not is_agent_config_shallowly_valid(system_config["agent"]):
            raise ValueError(f"The system configuration does not have a valid agent configuration: {system_config}")

        if not is_seekers_config_shallowly_valid(system_config["seekers"]):
            raise ValueError(f"The system configuration does not have valid seeker configurations: {system_config}")

        if not is_environment_config_shallowly_valid(system_config["environment"]):
            raise ValueError(f"The system configuration does not have a valid environment configuration: {system_config}")

        system_configuration = SystemConfiguration(
            agent = AgentConfiguration.from_dict(system_config["agent"]),
            seekers = [SeekerConfiguration.from_dict(seeker_config) for seeker_config in system_config["seekers"]],
            environment = EnvironmentConfiguration.from_dict(system_config["environment"]),
            num_seekers = len(system_config["seekers"])
        )
        validate_system_configuration(system_configuration)

        return system_configuration

    @staticmethod
    def parse_config_file(config_file: Path) -> "SystemConfiguration":
        """
        Parses a YAML configuration file and returns a SystemConfiguration object.
        
        :param config_file: The path to the YAML configuration file.
        :return: A SystemConfiguration object.
        """
        if not is_yaml_file(config_file):
            raise ValueError(f"The specified configuration file must be a valid YAML file: {config_file}")

        data = load_yaml_file(config_file)

        if not is_system_config_shallowly_valid(data):
            raise ValueError(f"The specified configuration file is not a valid system configuration: {config_file}")

        return SystemConfiguration.from_dict(data)
