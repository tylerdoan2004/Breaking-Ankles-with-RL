"""
This module provides helper functions for the reactive avoidance environment.
"""
import numpy as np
from collections import deque
from random import choice
from typing import Iterable, Literal
from gymnasium.spaces import Box
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.grid_dimensions import GridDimensions
from src.utils.environment.vector import Vector


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
    # At some time step, the agent observes four channels of information: whether the agent cannot observe a cell, whether the agent observes an out-of-bounds cell, whether the agent observes an obstacle, and whether the agent observes a seeker
    observation_channels = 4
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


def compute_offset_to_line_mapping(radius: int) -> dict[Vector, list[Coordinates]]:
    """
    Computes a mapping from each offset vector within a Chebyshev ball of the given radius to the discrete line segment from the origin to that offset.
    
    :param radius: The radius of the Chebyshev ball.
    :return: A mapping from each offset vector within a Chebyshev ball of the given radius to the discrete line segment from the origin to that offset.
    """
    return {Vector(dx, dy): bresenham_line(Coordinates(0, 0), Coordinates(dx, dy)) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)}


def can_see(line_segment: list[Coordinates], obstacles: Iterable[Coordinates]) -> bool:
    """
    Checks if the end of a line segment is visible from the start of the line segment given a set of obstacles' coordinates.
    
    :param line_segment: The line segment to check.
    :param obstacles: The obstacles' coordinates.
    :return: True if the end of the line segment is visible from the start of the line segment, False otherwise.
    """
    for coordinates in line_segment[1:-1]:
        if coordinates in obstacles:
            return False
    return True


def bresenham_line(start: Coordinates, end: Coordinates) -> list[Coordinates]:
    """
    Generates a list of coordinates representing a line segment using the Bresenham algorithm.
    
    :param start: The starting coordinates of the line segment.
    :param end: The ending coordinates of the line segment.
    :return: A list of coordinates representing the line segment.
    """
    # NOTE: This Bresenham algorithm implementation permits diagonal lines
    dx = abs(end.x - start.x)
    dy = -abs(end.y - start.y)
    sign_x = 1 if start.x < end.x else -1
    sign_y = 1 if start.y < end.y else -1
    error = dx + dy

    line_coordinates = []
    current_coordinates = start
    while True:
        line_coordinates.append(current_coordinates)
        if current_coordinates == end:
            break
        doubled_error = error * 2
        if doubled_error >= dy:
            error += dy
            current_coordinates += Vector(sign_x, 0)
        if doubled_error <= dx:
            error += dx
            current_coordinates += Vector(0, sign_y)
    return line_coordinates


def can_entity_be_in(coordinates: Coordinates, *, grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], seekers_coordinates: Iterable[Coordinates]) -> bool:
    """
    Checks if an entity can be in the given coordinates.
    
    :param coordinates: The coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :return: True if the entity can be in the given coordinates, False otherwise.
    """
    if not grid_dimensions.contains_coordinates(coordinates):
        return False
    if coordinates in obstacles_coordinates:
        return False
    if coordinates in seekers_coordinates:
        return False
    return True


# TODO: Use this function for validation logic
def move_agent(*,
               current_coordinates: Coordinates, velocity: int, movement_vector: Vector,
               grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], seekers_coordinates: Iterable[Coordinates],
               goal_coordinates: Coordinates) -> tuple[Coordinates, bool, bool]:
    """
    Moves the agent in the environment based on its current coordinates, velocity, and action.
    
    :param current_coordinates: The current coordinates of the agent.
    :param velocity: The velocity of the agent.
    :param movement_vector: The movement vector associated with the agent's action.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: A tuple containing the final coordinates of the agent, whether the agent collided (out-of-bounds, obstacle, or seeker), and whether the agent reached the goal.
    """
    # NOTE: This movement allows the agent to slip through diagonal obstacles
    intermediate_coordinates = current_coordinates
    for _ in range(velocity):
        intermediate_coordinates += movement_vector
        if not can_entity_be_in(intermediate_coordinates, grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = seekers_coordinates):
            return intermediate_coordinates, True, False
        if intermediate_coordinates == goal_coordinates:
            return intermediate_coordinates, False, True
    return intermediate_coordinates, False, False


def calculate_seeker_legal_moves(*,
                                 current_coordinates: Coordinates,
                                 grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], seekers_coordinates: Iterable[Coordinates],
                                 goal_coordinates: Coordinates) -> list[Vector]:
    """
    Computes the legal moves for a seeker.
    
    :param current_coordinates: The current coordinates of the seeker.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: A list of legal moves for a seeker.
    """
    # The seeker may always choose not to move
    legal_moves = [Vector(0, 0)]
    for movement_vector in (Vector(0, 1), Vector(1, 0), Vector(0, -1), Vector(-1, 0), Vector(1, 1), Vector(1, -1), Vector(-1, 1), Vector(-1, -1)):
        # Seekers may not collide with obstacles, other seekers, or out-of-bounds coordinates
        if not can_entity_be_in(current_coordinates + movement_vector,
                                grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = seekers_coordinates):
            continue
        # Seekers may not collide with the goal
        if current_coordinates + movement_vector == goal_coordinates:
            continue
        legal_moves.append(movement_vector)
    return legal_moves


def chebyshev_distance(first_coordinates: Coordinates, second_coordinates: Coordinates) -> int:
    """
    Computes the Chebychev distance between two coordinates.
    
    :param first_coordinates: The first pair of coordinates.
    :param second_coordinates: The second pair of coordinates.
    :return: The Chebychev distance between the two coordinates.
    """
    return max(abs(first_coordinates.x - second_coordinates.x), abs(first_coordinates.y - second_coordinates.y))


def greedy_seeker_policy(legal_moves: Iterable[Vector], *,
                         current_coordinates: Coordinates,
                         current_agent_coordinates: Coordinates) -> Vector:
    """
    Computes the legal move that minimizes the Chebychev distance to the agent. Breaks ties uniformly at random.
    
    :param legal_moves: The legal moves for a seeker.
    :param current_coordinates: The current coordinates of the seeker.
    :param current_agent_coordinates: The current coordinates of the agent.
    :return: A legal move that minimizes the Chebychev distance to the agent.
    """
    legal_moves = list(legal_moves)
    minimum_distance = min(chebyshev_distance(current_coordinates + movement_vector, current_agent_coordinates) for movement_vector in legal_moves)
    candidate_moves = [movement_vector for movement_vector in legal_moves if chebyshev_distance(current_coordinates + movement_vector, current_agent_coordinates) == minimum_distance]
    return choice(candidate_moves)


def seeker_policy(policy: Literal["random", "greedy"], *,
                  current_coordinates: Coordinates, current_agent_coordinates: Coordinates,
                  grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], seekers_coordinates: Iterable[Coordinates],
                  goal_coordinates: Coordinates) -> Vector:
    """
    Chooses a movement vector for a seeker.
    
    :param policy: The policy to use for choosing the movement vector.
    :param current_coordinates: The current coordinates of the seeker.
    :param current_agent_coordinates: The current coordinates of the agent.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: A movement vector for a seeker.
    """
    legal_moves = calculate_seeker_legal_moves(current_coordinates = current_coordinates,
                                               grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = seekers_coordinates, goal_coordinates = goal_coordinates)
    if not legal_moves:
        return Vector(0, 0)
    if policy == "random":
        return choice(legal_moves)
    if policy == "greedy":
        return greedy_seeker_policy(legal_moves,
                                    current_coordinates = current_coordinates,
                                    current_agent_coordinates = current_agent_coordinates)
    raise ValueError(f"Unsupported seeker policy: {policy}")


def move_seeker(*,
                current_coordinates: Coordinates, velocity: int,
                policy: Literal["random", "greedy"] = "greedy", current_agent_coordinates: Coordinates,
                grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], seekers_coordinates: Iterable[Coordinates], goal_coordinates: Coordinates) -> Coordinates:
    """
    Moves the seeker in the environment based on its current coordinates, velocity, and policy.
    
    :param current_coordinates: The current coordinates of the seeker.
    :param velocity: The velocity of the seeker.
    :param policy: The policy to use for choosing the movement vector.
    :param current_agent_coordinates: The current coordinates of the agent.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: The final coordinates of the seeker.
    """
    intermediate_coordinates = current_coordinates
    for _ in range(velocity):
        movement_vector = seeker_policy(policy,
                                        current_coordinates = intermediate_coordinates, current_agent_coordinates = current_agent_coordinates,
                                        grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = seekers_coordinates, goal_coordinates = goal_coordinates)
        intermediate_coordinates += movement_vector
    return intermediate_coordinates


def compute_time_steps_to_goal_mapping(*, goal_coordinates: Coordinates, grid_dimensions: GridDimensions, obstacles_coordinates: Iterable[Coordinates], agent_velocity: int) -> dict[Coordinates, int]:
    """
    Computes the minimum number of time steps to reach the goal coordinates for each valid pair of coordinates in the grid.
    
    :return: A dictionary mapping each valid pair of coordinates in the grid to the minimum number of time steps to reach the goal coordinates from that pair of coordinates.
    """
    obstacles_coordinates = frozenset(obstacles_coordinates)
    # Determine all valid coordinates in the grid
    valid_coordinates: list[Coordinates] = []
    for x in range(grid_dimensions.width):
        for y in range(grid_dimensions.height):
            coordinates = Coordinates(x, y)
            if not can_entity_be_in(coordinates, grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = ()):
                continue
            valid_coordinates.append(coordinates)

    # Build predecessor lists
    predecessors: dict[Coordinates, list[Coordinates]] = {coordinates: [] for coordinates in valid_coordinates}
    if goal_coordinates not in predecessors:
        return {}
    for current_coordinates in valid_coordinates:
        for movement_vector in (Vector(0, 1), Vector(1, 0), Vector(0, -1), Vector(-1, 0), Vector(1, 1), Vector(1, -1), Vector(-1, 1), Vector(-1, -1)):
            next_coordinates, collided, _ = move_agent(current_coordinates = current_coordinates, velocity = agent_velocity, movement_vector = movement_vector,
                                                       grid_dimensions = grid_dimensions, obstacles_coordinates = obstacles_coordinates, seekers_coordinates = (), goal_coordinates = goal_coordinates)
            if collided:
                continue
            if next_coordinates in predecessors:
                predecessors[next_coordinates].append(current_coordinates)

    # Reverse BFS from the goal coordinates
    time_steps_to_goal = {goal_coordinates: 0}
    queue = deque([goal_coordinates])
    while queue:
        current_coordinates = queue.popleft()
        current_time_steps_to_goal = time_steps_to_goal[current_coordinates]
        for predecessor_coordinates in predecessors[current_coordinates]:
            if predecessor_coordinates in time_steps_to_goal:
                continue
            time_steps_to_goal[predecessor_coordinates] = current_time_steps_to_goal + 1
            queue.append(predecessor_coordinates)
    return time_steps_to_goal
