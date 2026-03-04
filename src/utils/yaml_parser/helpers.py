"""
This module provides helper functions for the YAML parser.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils.yaml_parser.configuration import SystemConfiguration


import yaml

from collections import deque
from math import ceil
from pathlib import Path
from typing import Any, Optional
from warnings import warn
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.grid_dimensions import GridDimensions


USAGE_STRING = "Usage: python3 main.py [CONFIG_FILE].yaml"
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, -1), (-1, 1)]


def print_usage() -> None:
    """
    Prints the usage string for the program.

    :return: None.
    """
    print(USAGE_STRING)
    exit(1)


def is_yaml_file(file_path: Path) -> bool:
    """
    Checks if the specified file is a valid YAML file.

    :param file_path: The path to the file to check.
    :return: True if the file is a valid YAML file, False otherwise.
    """
    return file_path.is_file() and file_path.suffix == ".yaml"


def load_yaml_file(file_path: Path) -> Any:
    """
    Loads a YAML file and returns its contents.

    :param file_path: The path to the YAML file to load.
    :return: The contents of the YAML file.
    """
    try:
        with open(file_path, "r", encoding = "utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {file_path}")
        print(f"Encountered error: {e}")
        exit(1)


def is_config_version_valid(version: int) -> bool:
    """
    Checks if the specified config version is valid.

    :param version: The version to check.
    :return: True if the version is valid, False otherwise.
    """
    return isinstance(version, int) and version == 1


def is_system_config_shallowly_valid(data: Any) -> bool:
    """
    Checks shallowly if the object is a valid system configuration dictionary. Does not check if the agent configuration, seeker configurations, and environment configuration are valid.
    
    :param data: The object to check.
    :return: True if the object is a shallowlly valid system configuration, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"version", "agent", "seekers", "environment"}:
        return False
    if not is_config_version_valid(data["version"]):
        return False
    return True


def is_velocity_valid(velocity: int) -> bool:
    """
    Checks if the specified velocity is valid.

    :param velocity: The velocity to check.
    :return: True if the velocity is valid, False otherwise.
    """
    return isinstance(velocity, int) and velocity > 0


def is_visibility_radius_valid(visibility_radius: int) -> bool:
    """
    Checks if the specified visibility radius is valid.

    :param visibility_radius: The visibility radius to check.
    :return: True if the visibility radius is valid, False otherwise.
    """
    return isinstance(visibility_radius, int) and visibility_radius > 0


def is_observation_stack_depth_valid(observation_stack_depth: int) -> bool:
    """
    Checks if the specified observation stack depth is valid.
    
    :param observation_stack_depth: The observation stack depth to check.
    :return: True if the observation stack depth is valid, False otherwise.
    """
    # The observation stack includes the current observation; thus, the observation stack depth must be at least 1
    return isinstance(observation_stack_depth, int) and observation_stack_depth > 0


def is_agent_config_shallowly_valid(data: Any) -> bool:
    """
    Checks shallowly if the object is a valid agent configuration dictionary. Does not check if the start coordinates or goal coordinates are valid.
    
    :param data: The object to check.
    :return: True if the object is a shallowly valid agent configuration, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"start_coordinates", "goal_coordinates", "velocity", "visibility_radius", "observation_stack_depth"}:
        return False
    if not is_velocity_valid(data["velocity"]):
        return False
    if not is_visibility_radius_valid(data["visibility_radius"]):
        return False
    if not is_observation_stack_depth_valid(data["observation_stack_depth"]):
        return False
    return True


def is_coordinate_component_valid(component: Any) -> bool:
    """
    Checks if the specified coordinate component is valid.
    
    :param component: The coordinate component to check.
    :return: True if the coordinate component is valid, False otherwise.
    """
    return isinstance(component, int) and component >= 0


def is_coordinates_data_valid(data: Any) -> bool:
    """
    Checks if the specified coordinates data is valid.
    
    :param data: The coordinates data to check.
    :return: True if the coordinates data is valid, False otherwise.
    """
    if not isinstance(data, list):
        return False
    if len(data) != 2:
        return False
    if any(not is_coordinate_component_valid(component) for component in data):
        return False
    return True


def is_seeker_config_shallowly_valid(data: Any) -> bool:
    """
    Checks shallowly if the object is a valid seeker configuration dictionary. Does not check if the start coordinates are valid.
    
    :param data: The object to check.
    :return: True if the object is a shallowly valid seeker configuration, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"start_coordinates", "velocity"}:
        return False
    if not is_velocity_valid(data["velocity"]):
        return False
    return True


def is_seekers_config_shallowly_valid(data: Any) -> bool:
    """
    Checks shallowly if the object is a valid list of seeker configurations. Does not check if the start coordinates are valid.
    
    :param data: The object to check.
    :return: True if the object is a shallowly valid list of seeker configurations, False otherwise.
    """
    if not isinstance(data, list):
        return False
    if not data:
        return False
    if any(not is_seeker_config_shallowly_valid(seeker_config) for seeker_config in data):
        return False
    return True


def is_episode_time_limit_valid(episode_time_limit: int) -> bool:
    """
    Checks if the specified episode time limit is valid.
    
    :param episode_time_limit: The episode time limit to check.
    :return: True if the episode time limit is valid, False otherwise.
    """
    return isinstance(episode_time_limit, int) and episode_time_limit > 0


def is_dimension_size_valid(dimension_size: int) -> bool:
    """
    Checks if the specified dimension size is valid.
    
    :param dimension_size: The dimension size to check.
    :return: True if the dimension size is valid, False otherwise.
    """
    return isinstance(dimension_size, int) and dimension_size > 0


def is_grid_dimensions_data_valid(data: Any) -> bool:
    """
    Checks if the specified grid diemensions data is valid.
    
    :param data: The grid dimensions data to check.
    :return: True if the grid dimensions data is valid, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"width", "height"}:
        return False
    if any(not is_dimension_size_valid(dimension_size) for dimension_size in data.values()):
        return False
    return True


def is_obstacles_coordinates_data_valid(data: Any) -> bool:
    """
    Checks if the specified obstacles' coordinates data is valid.
    
    :param data: The obstacles' coordinates data to check.
    :return: True if the obstacles' coordinates data is valid, False otherwise.
    """
    if not isinstance(data, list):
        return False
    if any(not is_coordinates_data_valid(coordinates) for coordinates in data):
        return False
    return True


def is_environment_config_shallowly_valid(data: Any) -> bool:
    """
    Checks shallowly if the object is a valid environment configuration dictionary. Does not check if the grid dimensions or the obstacles' coordinates are valid.
    
    :param data: The object to check.
    :return: True if the object is a shallowly valid environment configuration, False otherwise.
    """
    if not isinstance(data, dict):
        return False
    if set(data.keys()) != {"grid_dimensions", "obstacles_coordinates", "episode_time_limit"}:
        return False
    if not is_episode_time_limit_valid(data["episode_time_limit"]):
        return False
    return True


def are_obstacles_unique(obstacles_cordinates: list[Coordinates]) -> bool:
    """
    Checks if the obstacles' coordinates are unique.
    
    :param obstacles_cordinates: The obstacles' coordinates to check.
    :return: True if the obstacles' coordinates are unique, False otherwise.
    """
    return len(set(obstacles_cordinates)) == len(obstacles_cordinates)


def are_obstacles_in_grid(obstacles_coordinates: list[Coordinates], grid_dimensions: GridDimensions) -> bool:
    """
    Checks if the obstacles' coordinates are in the grid dimensions.
    
    :param obstacles_coordinates: The obstacles' coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :return: True if the obstacles' coordinates are in the grid dimensions, False otherwise.
    """
    if any(not grid_dimensions.contains_coordinates(obstacle_coordinates) for obstacle_coordinates in obstacles_coordinates):
        return False
    return True


def does_agent_start_in_grid(start_coordinates: Coordinates, grid_dimensions: GridDimensions) -> bool:
    """
    Checks if the agent's start coordinates are in the grid dimensions.
    
    :param start_coordinates: The agent's start coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :return: True if the agent's start coordinates are in the grid dimensions, False otherwise.
    """
    return grid_dimensions.contains_coordinates(start_coordinates)


def is_agent_goal_in_grid(goal_coordinates: Coordinates, grid_dimensions: GridDimensions) -> bool:
    """
    Checks if the agent's goal coordinates are in the grid dimensions.
    
    :param goal_coordinates: The agent's goal coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :return: True if the agent's goal coordinates are in the grid dimensions, False otherwise.
    """
    return grid_dimensions.contains_coordinates(goal_coordinates)


def are_seekers_unique(seekers_start_coordinates: list[Coordinates]) -> bool:
    """
    Checks if the seekers' start coordinates are unique.
    
    :param seekers_start_coordinates: The seekers' start coordinates to check.
    :return: True if the seekers' start coordinates are unique, False otherwise.
    """
    return len(set(seekers_start_coordinates)) == len(seekers_start_coordinates)


def do_seekers_start_in_grid(seekers_start_coordinates: list[Coordinates], grid_dimensions: GridDimensions) -> bool:
    """
    Checks if the seekers' start coordinates are in the grid dimensions.
    
    :param seekers_start_coordinates: The seekers' start coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :return: True if the seekers' start coordinates are in the grid dimensions, False otherwise.
    """
    if any(not grid_dimensions.contains_coordinates(seeker_start_coordinates) for seeker_start_coordinates in seekers_start_coordinates):
        return False
    return True


def intersects(first_coordinates_list: list[Coordinates], second_coordinates_list: list[Coordinates]) -> bool:
    """
    Checks if the specified lists of coordinates intersect.
    
    :param first_coordinates_list: The first list of coordinates to check.
    :param second_coordinates_list: The second list of coordinates to check.
    :return: True if the lists of coordinates intersect, False otherwise.
    """
    return not set(first_coordinates_list).isdisjoint(second_coordinates_list)


def does_agent_velocity_exceed_grid_dimensions(velocity: int, grid_dimensions: GridDimensions) -> bool:
    """
    Checks if the agent's velocity exceeds the grid dimensions.
    
    :param velocity: The agent's velocity to check.
    :param grid_dimensions: The grid dimensions.
    :return: True if the agent's velocity exceeds the grid dimensions, False otherwise.
    """
    return velocity > min(grid_dimensions.width, grid_dimensions.height)


def does_agent_velocity_exceed_visibility_radius(velocity: int, visibility_radius: int) -> bool:
    """
    Checks if the agent's velocity exceeds the agent's visibility radius.
    
    :param velocity: The agent's velocity to check.
    :param visibility_radius: The agent's visibility radius.
    :return: True if the agent's velocity exceeds the agent's visibility radius, False otherwise.
    """
    return velocity > visibility_radius


def minimum_unit_moves_to_reach_goal(grid_dimensions: GridDimensions,
                                     start_coordinates: Coordinates, goal_coordinates: Coordinates,
                                     obstacles_coordinates: list[Coordinates]) -> Optional[int]:
    """
    Calculates the minimum number of unit moves from the agent's start coordinates to the agent's goal coordinates without colliding with obstacles. A unit move is a unit of motion in any of the cardinal or diagonal directions. This function implements the BFS algorithm.
    
    :param grid_dimensions: The grid dimensions.
    :param start_coordinates: The start coordinates.
    :param goal_coordinates: The goal coordinates.
    :param obstacles_coordinates: The obstacles' coordinates.
    :return: The minimum number of moves or None if no path exists.
    """
    if goal_coordinates in obstacles_coordinates:
        return None
    # Because we perform goal checking when expanding each node, we must goal check the start coordinates
    if start_coordinates == goal_coordinates:
        return 0

    queue = deque([(start_coordinates, 0)])
    # Because we have already goal checked the start coordinates, we can add the start coordinates to the visited set
    visited = {start_coordinates}
    while queue:
        # Dequeue the current node from the queue
        current_coordinates, moves = queue.popleft()
        # Expand the current node
        for direction in DIRECTIONS:
            next_coordinates = Coordinates(current_coordinates.x + direction[0], current_coordinates.y + direction[1])
            # If the next node is invalid, skip the next node
            if not grid_dimensions.contains_coordinates(next_coordinates) or next_coordinates in obstacles_coordinates:
                continue
            # If the next node has already been visited, skip the next node
            if next_coordinates in visited:
                continue
            # Goal check the next node
            if next_coordinates == goal_coordinates:
                return moves + 1
            # If the goal check fails, add the next node to the visited set and enqueue the next node
            visited.add(next_coordinates)
            queue.append((next_coordinates, moves + 1))
    return None


# TODO: Fix minimum time steps calculation
def calculate_minimum_time_steps_to_reach_goal(minimum_moves: int, velocity: int) -> int:
    """
    Calculates the minimum number of time steps from the agent's start coordinates to the agent's goal coordinates without colliding with obstacles. A time step is a unit of time in the environment.
    
    :param minimum_moves: The minimum number of moves from the agent's start coordinates to the agent's goal coordinates without colliding with obstacles.
    :param velocity: The agent's velocity.
    :return: The minimum number of time steps.
    """
    if velocity <= 0:
        raise ValueError("The agent's velocity must be positive.")
    return ceil(minimum_moves / velocity)


def is_episode_time_limit_sufficient(episode_time_limit: int, minimum_time_steps: int) -> bool:
    """
    Checks if the episode time limit is sufficient.
    
    :param episode_time_limit: The episode time limit.
    :param minimum_time_steps: The minimum time steps.
    :return: True if the episode time limit is sufficient, False otherwise.
    """
    return episode_time_limit >= minimum_time_steps


# TODO: Fix episode time limit validation
# TODO: Validate the location of the agent, seekers, obstacles, and goal with respect to the walls of the grid
def validate_system_configuration(system_configuration: SystemConfiguration) -> None:
    """
    Validates the system configuration.
    
    :param system_configuration: The system configuration to validate.
    :return: None
    """
    grid_dimensions = system_configuration.environment.grid_dimensions

    if not are_obstacles_unique(system_configuration.environment.obstacles_coordinates):
        raise ValueError(f"The obstacles' coordinates are not unique: {system_configuration.environment.obstacles_coordinates}")
    if not are_obstacles_in_grid(system_configuration.environment.obstacles_coordinates, grid_dimensions):
        raise ValueError(f"The obstacles' coordinates are not in the grid dimensions: {system_configuration.environment.obstacles_coordinates}")

    if not does_agent_start_in_grid(system_configuration.agent.start_coordinates, grid_dimensions):
        raise ValueError(f"The agent's start coordinates are not in the grid dimensions: {system_configuration.agent.start_coordinates}")
    if not is_agent_goal_in_grid(system_configuration.agent.goal_coordinates, grid_dimensions):
        raise ValueError(f"The agent's goal coordinates are not in the grid dimensions: {system_configuration.agent.goal_coordinates}")

    seekers_start_coordinates = [seeker.start_coordinates for seeker in system_configuration.seekers]
    if not are_seekers_unique(seekers_start_coordinates):
        raise ValueError(f"The seekers' start coordinates are not unique: {seekers_start_coordinates}")
    if not do_seekers_start_in_grid(seekers_start_coordinates, grid_dimensions):
        raise ValueError(f"The seekers' start coordinates are not in the grid dimensions: {seekers_start_coordinates}")

    if intersects([system_configuration.agent.start_coordinates], system_configuration.environment.obstacles_coordinates):
        raise ValueError(f"The agent's start coordinates intersect with an obstacle: {system_configuration.agent.start_coordinates}")
    if intersects([system_configuration.agent.start_coordinates], [system_configuration.agent.goal_coordinates]):
        raise ValueError(f"The agent's start coordinates intersect with the agent's goal coordinates: {system_configuration.agent.start_coordinates}")
    if intersects([system_configuration.agent.start_coordinates], seekers_start_coordinates):
        raise ValueError(f"The agent's start coordinates intersect with a seeker's start coordinates: {system_configuration.agent.start_coordinates}")
    if intersects([system_configuration.agent.goal_coordinates], system_configuration.environment.obstacles_coordinates):
        raise ValueError(f"The agent's goal coordinates intersect with an obstacle: {system_configuration.agent.goal_coordinates}")
    if intersects([system_configuration.agent.goal_coordinates], seekers_start_coordinates):
        raise ValueError(f"The agent's goal coordinates intersect with a seeker's start coordinates: {system_configuration.agent.goal_coordinates}")
    if intersects(seekers_start_coordinates, system_configuration.environment.obstacles_coordinates):
        raise ValueError(f"The seekers' start coordinates intersect with an obstacle: {seekers_start_coordinates}")

    if does_agent_velocity_exceed_grid_dimensions(system_configuration.agent.velocity, grid_dimensions):
        raise ValueError(f"The agent's velocity exceeds the grid dimensions: {system_configuration.agent.velocity}")
    if does_agent_velocity_exceed_visibility_radius(system_configuration.agent.velocity, system_configuration.agent.visibility_radius):
        warn(f"The agent's velocity exceeds the agent's visibility radius: {system_configuration.agent.velocity}", RuntimeWarning)

    minimum_moves = minimum_unit_moves_to_reach_goal(grid_dimensions,
                                                system_configuration.agent.start_coordinates, system_configuration.agent.goal_coordinates,
                                                system_configuration.environment.obstacles_coordinates)
    if minimum_moves is None:
        raise ValueError(f"The agent cannot reach the goal coordinates from the start coordinates given the obstacles: {system_configuration.environment.obstacles_coordinates}")

    minimum_time_steps_to_reach_goal = calculate_minimum_time_steps_to_reach_goal(minimum_moves, system_configuration.agent.velocity)
    if not is_episode_time_limit_sufficient(system_configuration.environment.episode_time_limit, minimum_time_steps_to_reach_goal):
        raise ValueError(f"The agent cannot reach the goal coordinates within the episode time limit: {system_configuration.environment.episode_time_limit}")
