"""
This module provides helper functions for the reactive avoidance environment.
"""
import numpy as np
from gymnasium.spaces import Box
from utils.environment.coordinates import Coordinates
from utils.environment.vector import Vector


def calculate_observation_space(visibility_radius: int, observation_stack_depth: int) -> Box:
    """
    Calculates the observation space for the reactive avoidance environment. Each of the agent's observations of the environment state is a fixed-size encoding of the agent's current coordinates, the relative coordinates of the goal, the agent's local observation of the environment state (consisting of whether the agent observes an obstacle, whether the agent observes a seeker, and whether the agent observes the goal), and the agent's previous local observations of the environment state.
    
    :param visibility_radius: The agent's visibility radius.
    :param observation_stack_depth: The number of consecutive observations to include in the agent's observation of the environment state (including the current observation).
    :return: The observation space for the reactive avoidance environment.
    """
    # Given the environment state at some time step, we encode the agent's position (a two-dimensional vector) and the relative position of the goal (a two-dimensional vector)
    current_observation_dimensions = 4
    # Calculate the size of the area visible to the agent
    visible_area_size = (2 * visibility_radius + 1) ** 2
    # At some time step, the agent observes three channels of information: whether the agent observes an obstacle, whether the agent observes a seeker, and whether the agent observes the goal
    observation_channels = 3
    # Given the environment state at some time step, we encode the agent's three channels of information for each cell in the visible area (referred to as the agent's local observation of the environment state)
    dimensions_per_local_observation = observation_channels * visible_area_size
    # Given the environment state at some time step, we additionally encode the agent's previous observations of the environment state (excluding the agent's previous positions and the previous relative positions of the goal); that is, we additionally encode the agent's previous local observations of the environment state
    observation_stack_dimensions = observation_stack_depth * dimensions_per_local_observation
    # Given the environment state at some time step, we encode, in total, the agent's position, the relative position of the goal, the relative positions of visible obstacles and seekers, and the agent's previous observations of the environment state
    observation_dimensions = current_observation_dimensions + observation_stack_dimensions
    return Box(low = -1, high = 1, shape = (observation_dimensions,), dtype = np.float32)

def scale_absolute_coordinates(coordinates: Coordinates, grid_width: int, grid_height: int) -> np.ndarray:
    """
    Scales absolute coordinates in a two-dimensional gridworld to a range of [-1, 1].
    
    :param coordinates: The absolute coordinates to scale. Each coordinate lies between [0, grid_width - 1].
    :param grid_width: The width of the gridworld.
    :param grid_height: The height of the gridworld.
    :return: The scaled coordinates.
    """
    scaled_x_coordinate = 2.0 * (coordinates.x / max(1, grid_width - 1)) - 1.0
    scaled_y_coordinate = 2.0 * (coordinates.y / max(1, grid_height - 1)) - 1.0
    return np.array([scaled_x_coordinate, scaled_y_coordinate], dtype = np.float32)

def scale_relative_vector(vector: Vector, grid_width: int, grid_height: int) -> np.ndarray:
    """
    Scales a relative vector in a two-dimensional gridworld to a range of [-1, 1].
    
    :param vector: The relative vector to scale. Each component lies between [-(grid_width - 1), grid_width - 1].
    :param grid_width: The width of the gridworld.
    :param grid_height: The height of the gridworld.
    :return: The scaled relative vector.
    """
    scaled_x_component = vector.x / max(1, grid_width - 1)
    scaled_y_component = vector.y / max(1, grid_height - 1)
    return np.array([scaled_x_component, scaled_y_component], dtype = np.float32)
