"""
This module provides helper functions for the reactive avoidance environment.
"""
import numpy as np
from collections import deque
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from heapq import heappop, heappush
from random import choice
from typing import Callable, Iterable, Literal, Optional
from gymnasium.spaces import Box
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.grid_dimensions import GridDimensions
from src.utils.environment.vector import Vector


MOVEMENT_VECTORS = (
    Vector(0, 0),
    Vector(0, 1),
    Vector(1, 0),
    Vector(0, -1),
    Vector(-1, 0),
    Vector(1, 1),
    Vector(1, -1),
    Vector(-1, 1),
    Vector(-1, -1)
)


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


def can_entity_be_in(entity_type: Literal["agent", "seeker"], coordinates: Coordinates, *,
                     grid_dimensions: GridDimensions,
                     obstacles_coordinates: AbstractSet[Coordinates],
                     seekers_coordinates: AbstractSet[Coordinates],
                     goal_coordinates: Coordinates) -> bool:
    """
    Checks if an entity can be in the given coordinates.
    
    :param entity_type: The type of entity. Must be "agent" or "seeker".
    :param coordinates: The coordinates to check.
    :param grid_dimensions: The grid dimensions.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: True if the entity can be in the given coordinates, False otherwise.
    """
    if not grid_dimensions.contains_coordinates(coordinates):
        return False
    if coordinates in obstacles_coordinates:
        return False
    if coordinates in seekers_coordinates:
        return False
    if entity_type == "seeker" and coordinates == goal_coordinates:
        return False
    return True


# TODO: Use this function for validation logic
def move_agent(*,
               current_coordinates: Coordinates, velocity: int, movement_vector: Vector,
               grid_dimensions: GridDimensions,
               obstacles_coordinates: AbstractSet[Coordinates],
               seekers_coordinates: AbstractSet[Coordinates],
               goal_coordinates: Coordinates) -> tuple[Coordinates, Coordinates, bool, bool]:
    """
    Moves the agent in the environment based on its current coordinates, velocity, and action.
    
    :param current_coordinates: The current coordinates of the agent.
    :param velocity: The velocity of the agent.
    :param movement_vector: The movement vector associated with the agent's action.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: A tuple containing the updated coordinates of the agent, the last valid coordinates of the agent, whether the agent collided (out-of-bounds, obstacle, or seeker), and whether the agent reached the goal.
    """
    # NOTE: This movement allows the agent to slip through diagonal obstacles
    last_valid_coordinates = current_coordinates
    intermediate_coordinates = current_coordinates
    for _ in range(velocity):
        intermediate_coordinates += movement_vector
        if not can_entity_be_in("agent", intermediate_coordinates,
                                grid_dimensions = grid_dimensions,
                                obstacles_coordinates = obstacles_coordinates,
                                seekers_coordinates = seekers_coordinates,
                                goal_coordinates = goal_coordinates):
            return intermediate_coordinates, last_valid_coordinates, True, False
        last_valid_coordinates = intermediate_coordinates
        if intermediate_coordinates == goal_coordinates:
            return intermediate_coordinates, last_valid_coordinates, False, True
    return intermediate_coordinates, last_valid_coordinates, False, False


def calculate_entity_legal_neighboring_coordinates(*,
                                                   entity_type: Literal["agent", "seeker"],
                                                   current_coordinates: Coordinates,
                                                   grid_dimensions: GridDimensions,
                                                   obstacles_coordinates: AbstractSet[Coordinates],
                                                   seekers_coordinates: AbstractSet[Coordinates],
                                                   goal_coordinates: Coordinates,
                                                   exclude_current_coordinates: bool = False) -> list[Coordinates]:
    """
    Computes the legal neighboring coordinates reachable by an entity in one substep.
    
    :param entity_type: The type of entity. Must be "agent" or "seeker".
    :param current_coordinates: The current coordinates of the entity.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :param exclude_current_coordinates: Whether to exclude the current coordinates from the legal neighboring coordinates.
    :return: A list of legal neighboring coordinates reachable by the entity in one substep.
    """
    legal_neighboring_coordinates = []
    for movement_vector in MOVEMENT_VECTORS:
        if exclude_current_coordinates and movement_vector == Vector(0, 0):
            continue
        neighboring_coordinates = current_coordinates + movement_vector
        # If the entity cannot be in the neighboring coordinates, the neighboring coordinates are not legal
        if not can_entity_be_in(entity_type, neighboring_coordinates,
                                grid_dimensions = grid_dimensions,
                                obstacles_coordinates = obstacles_coordinates,
                                seekers_coordinates = seekers_coordinates,
                                goal_coordinates = goal_coordinates):
            continue
        legal_neighboring_coordinates.append(neighboring_coordinates)
    return legal_neighboring_coordinates


def calculate_entity_legal_moves(*,
                                 entity_type: Literal["agent", "seeker"],
                                 current_coordinates: Coordinates,
                                 grid_dimensions: GridDimensions,
                                 obstacles_coordinates: AbstractSet[Coordinates],
                                 seekers_coordinates: AbstractSet[Coordinates],
                                 goal_coordinates: Coordinates) -> list[Vector]:
    """
    Computes the legal moves executable by an entity in one substep.
    
    :param entity_type: The type of entity. Must be "agent" or "seeker".
    :param current_coordinates: The current coordinates of the entity.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param seekers_coordinates: The seekers' coordinates.
    :param goal_coordinates: The goal coordinates.
    :return: A list of legal moves executable by the entity in one substep.
    """
    legal_neighboring_coordinates = calculate_entity_legal_neighboring_coordinates(entity_type = entity_type,
                                                                                   current_coordinates = current_coordinates,
                                                                                   grid_dimensions = grid_dimensions,
                                                                                   obstacles_coordinates = obstacles_coordinates,
                                                                                   seekers_coordinates = seekers_coordinates,
                                                                                   goal_coordinates = goal_coordinates)
    legal_moves: list[Vector] = []
    for neighboring_coordinates in legal_neighboring_coordinates:
        legal_moves.append(neighboring_coordinates - current_coordinates)
    return legal_moves


def chebyshev_distance(first_coordinates: Coordinates, second_coordinates: Coordinates) -> int:
    """
    Computes the Chebyshev distance between two coordinates.
    
    :param first_coordinates: The first pair of coordinates.
    :param second_coordinates: The second pair of coordinates.
    :return: The Chebyshev distance between the two coordinates.
    """
    return max(abs(first_coordinates.x - second_coordinates.x), abs(first_coordinates.y - second_coordinates.y))


def greedy_seeker_policy(legal_moves: Iterable[Vector], *,
                         current_coordinates: Coordinates,
                         current_agent_coordinates: Coordinates) -> Vector:
    """
    Computes the legal move that minimizes the Chebyshev distance to the agent. Breaks ties uniformly at random.
    
    :param legal_moves: The legal moves for a seeker.
    :param current_coordinates: The current coordinates of the seeker.
    :param current_agent_coordinates: The current coordinates of the agent.
    :return: A legal move that minimizes the Chebyshev distance to the agent.
    """
    legal_moves = list(legal_moves)
    minimum_distance = min(chebyshev_distance(current_coordinates + movement_vector, current_agent_coordinates)
                           for movement_vector in legal_moves)
    candidate_moves = [movement_vector for movement_vector in legal_moves
                       if chebyshev_distance(current_coordinates + movement_vector, current_agent_coordinates) == minimum_distance]
    return choice(candidate_moves)


def a_star_seeker_policy(legal_moves: Iterable[Vector], *,
                         current_coordinates: Coordinates,
                         current_agent_coordinates: Coordinates,
                         grid_dimensions: GridDimensions,
                         obstacles_coordinates: AbstractSet[Coordinates],
                         other_seekers_coordinates: AbstractSet[Coordinates],
                         goal_coordinates: Coordinates,
                         heuristic: Callable[[Coordinates, Coordinates], int | float] = chebyshev_distance) -> Vector:
    """
    Computes the legal move that begins the shortest path from the seeker's current coordinates to the agent's current coordinates using A* pathfinding. Breaks ties by heap order. If no path exists, the policy falls back to the greedy policy.
    
    :param legal_moves: The legal moves for a seeker.
    :param current_coordinates: The current coordinates of the seeker.
    :param current_agent_coordinates: The current coordinates of the agent.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param other_seekers_coordinates: The coordinates of other seekers.
    :param goal_coordinates: The goal coordinates.
    :param heuristic: The heuristic function for A* pathfinding. Defaults to the Chebyshev distance. The heuristic function should be admissible.
    :return: A legal move that begins the shortest path from the seeker's current coordinates to the agent's current coordinates using A* pathfinding.
    """
    if current_coordinates == current_agent_coordinates:
        return Vector(0, 0)

    legal_moves = list(legal_moves)
    if not legal_moves:
        return Vector(0, 0)

    frontier: list[tuple[int | float, int, Coordinates]] = []
    heappush(frontier, (heuristic(current_coordinates, current_agent_coordinates), 0, current_coordinates))
    g_mapping: dict[Coordinates, int] = {current_coordinates: 0}
    previous: dict[Coordinates, Optional[Coordinates]] = {current_coordinates: None}

    # Perform A* pathfinding to find the shortest path from the seeker's current coordinates to the agent's current coordinates
    while frontier:
        _, g_intermediate, intermediate_coordinates = heappop(frontier)
        # Perform goal checking on pop
        if intermediate_coordinates == current_agent_coordinates:
            break
        # If the intermediate coordinates can be reached by a shorter path, continue
        if g_intermediate > g_mapping.get(intermediate_coordinates, float("inf")):
            continue
        legal_neighboring_coordinates = calculate_entity_legal_neighboring_coordinates(entity_type = "seeker",
                                                                                       current_coordinates = intermediate_coordinates,
                                                                                       grid_dimensions = grid_dimensions,
                                                                                       obstacles_coordinates = obstacles_coordinates,
                                                                                       seekers_coordinates = other_seekers_coordinates,
                                                                                       goal_coordinates = goal_coordinates,
                                                                                       exclude_current_coordinates = True)
        for neighboring_coordinates in legal_neighboring_coordinates:
            g_neighboring = g_intermediate + 1
            if g_neighboring >= g_mapping.get(neighboring_coordinates, float("inf")):
                continue
            f_neighboring = g_neighboring + heuristic(neighboring_coordinates, current_agent_coordinates)
            heappush(frontier, (f_neighboring, g_neighboring, neighboring_coordinates))
            g_mapping[neighboring_coordinates] = g_neighboring
            previous[neighboring_coordinates] = intermediate_coordinates

    # If no path exists, fall back to the greedy policy
    if current_agent_coordinates not in g_mapping:
        return greedy_seeker_policy(legal_moves,
                                    current_coordinates = current_coordinates,
                                    current_agent_coordinates = current_agent_coordinates)

    # Reconstruct the shortest path from the seeker's current coordinates to the agent's current coordinates
    shortest_path: list[Coordinates] = []
    intermediate_coordinates = current_agent_coordinates
    while intermediate_coordinates is not None and intermediate_coordinates != current_coordinates:
        shortest_path.append(intermediate_coordinates)
        intermediate_coordinates = previous.get(intermediate_coordinates)
    shortest_path_move = shortest_path[-1] - current_coordinates

    # If the shortest path move is not a legal move (for some reason), fall back to the greedy policy
    if shortest_path_move not in legal_moves:
        return greedy_seeker_policy(legal_moves,
                                    current_coordinates = current_coordinates,
                                    current_agent_coordinates = current_agent_coordinates)

    return shortest_path_move


def seeker_policy(policy: Literal["random", "greedy", "a-star"], *,
                  current_coordinates: Coordinates, current_agent_coordinates: Coordinates,
                  grid_dimensions: GridDimensions,
                  obstacles_coordinates: AbstractSet[Coordinates],
                  other_seekers_coordinates: AbstractSet[Coordinates],
                  goal_coordinates: Coordinates) -> Vector:
    """
    Chooses a movement vector for a seeker based on the specified policy.
    
    :param policy: The policy to use for choosing the movement vector.
    :param current_coordinates: The current coordinates of the seeker.
    :param current_agent_coordinates: The current coordinates of the agent.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param other_seekers_coordinates: The coordinates of other seekers.
    :param goal_coordinates: The goal coordinates.
    :return: A movement vector for a seeker chosen based on the specified policy.
    """
    legal_moves = calculate_entity_legal_moves(entity_type = "seeker",
                                               current_coordinates = current_coordinates,
                                               grid_dimensions = grid_dimensions,
                                               obstacles_coordinates = obstacles_coordinates,
                                               seekers_coordinates = other_seekers_coordinates,
                                               goal_coordinates = goal_coordinates)
    if not legal_moves:
        return Vector(0, 0)
    if policy == "random":
        return choice(legal_moves)
    if policy == "greedy":
        return greedy_seeker_policy(legal_moves,
                                    current_coordinates = current_coordinates,
                                    current_agent_coordinates = current_agent_coordinates)
    if policy == "a-star":
        return a_star_seeker_policy(legal_moves,
                                    current_coordinates = current_coordinates,
                                    current_agent_coordinates = current_agent_coordinates,
                                    grid_dimensions = grid_dimensions,
                                    obstacles_coordinates = obstacles_coordinates,
                                    other_seekers_coordinates = other_seekers_coordinates,
                                    goal_coordinates = goal_coordinates)
    raise ValueError(f"Unsupported seeker policy: {policy}")


def move_seeker(*,
                current_coordinates: Coordinates, velocity: int,
                policy: Literal["random", "greedy", "a-star"] = "a-star",
                current_agent_coordinates: Coordinates,
                grid_dimensions: GridDimensions,
                obstacles_coordinates: AbstractSet[Coordinates],
                other_seekers_coordinates: AbstractSet[Coordinates],
                goal_coordinates: Coordinates) -> Coordinates:
    """
    Moves the seeker in the environment based on its current coordinates, velocity, and policy.
    
    :param current_coordinates: The current coordinates of the seeker.
    :param velocity: The velocity of the seeker.
    :param policy: The policy to use for choosing the movement vector.
    :param current_agent_coordinates: The current coordinates of the agent.
    :param grid_dimensions: The grid dimensions of the environment.
    :param obstacles_coordinates: The obstacles' coordinates.
    :param other_seekers_coordinates: The coordinates of other seekers.
    :param goal_coordinates: The goal coordinates.
    :return: The final coordinates of the seeker after moving.
    """
    intermediate_coordinates = current_coordinates
    for _ in range(velocity):
        movement_vector = seeker_policy(policy,
                                        current_coordinates = intermediate_coordinates,
                                        current_agent_coordinates = current_agent_coordinates,
                                        grid_dimensions = grid_dimensions,
                                        obstacles_coordinates = obstacles_coordinates,
                                        other_seekers_coordinates = other_seekers_coordinates,
                                        goal_coordinates = goal_coordinates)
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
            if not can_entity_be_in("agent",
                                    coordinates,
                                    grid_dimensions = grid_dimensions,
                                    obstacles_coordinates = obstacles_coordinates,
                                    seekers_coordinates = frozenset(),
                                    goal_coordinates = goal_coordinates):
                continue
            valid_coordinates.append(coordinates)

    # Build predecessor lists
    predecessors: dict[Coordinates, list[Coordinates]] = {coordinates: [] for coordinates in valid_coordinates}
    if goal_coordinates not in predecessors:
        return {}
    for current_coordinates in valid_coordinates:
        for movement_vector in MOVEMENT_VECTORS:
            if movement_vector == Vector(0, 0):
                continue
            next_coordinates, _, collided, _ = move_agent(current_coordinates = current_coordinates, velocity = agent_velocity, movement_vector = movement_vector,
                                                          grid_dimensions = grid_dimensions,
                                                          obstacles_coordinates = obstacles_coordinates,
                                                          seekers_coordinates = frozenset(),
                                                          goal_coordinates = goal_coordinates)
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


def compute_terminal_rewards(*,
                             goal: bool,
                             collided: bool,
                             intercepted: bool,
                             truncated: bool) -> float:
    """
    Computes the reward term associated with terminal environment states.
    
    :param goal: Whether the agent reached the goal.
    :param collided: Whether the agent collided with an obstacle, a seeker, or an out-of-bounds cell.
    :param intercepted: Whether the agent was intercepted by a seeker.
    :param truncated: Whether the episode was truncated.
    :return: The reward term associated with terminal environment states.
    """
    if goal:
        return 5.0
    if collided or intercepted or truncated:
        return -5.0
    return 0.0


def compute_progress_rewards(*,
                             previous_time_steps_to_goal: Optional[int] = None,
                             current_time_steps_to_goal: Optional[int] = None,
                             progress_coefficient: float = 0.05) -> float:
    """
    Computes the reward term associated with agent progress in the environment.
    
    :param previous_time_steps_to_goal: The previous minimum number of time steps to reach the goal.
    :param current_time_steps_to_goal: The current minimum number of time steps to reach the goal.
    :param progress_coefficient: The coefficient for the progress reward term.
    :return: The reward term associated with agent progress in the environment.
    """
    if previous_time_steps_to_goal is None or current_time_steps_to_goal is None:
        return 0.0
    return progress_coefficient * (previous_time_steps_to_goal - current_time_steps_to_goal)


def compute_danger_penalty(*,
                           current_agent_coordinates: Coordinates,
                           currently_visible_seekers_coordinates: Iterable[Coordinates],
                           visibility_radius: int,
                           distance_metric: Callable[[Coordinates, Coordinates], int | float] = chebyshev_distance,
                           danger_coefficient: float = 0.05) -> float:
    """
    Computes the penalty term associated with the agent's danger level in the environment.

    :param current_agent_coordinates: The current coordinates of the agent after the agent's move.
    :param currently_visible_seekers_coordinates: The coordinates of the seekers currently visible to the agent.
    :param visibility_radius: The visibility radius of the agent.
    :param distance_metric: The distance metric to use for computing the distance between the agent and a seeker.
    :param danger_coefficient: The coefficient for the danger penalty term.
    :return: The penalty term associated with the agent's danger level in the environment.
    """
    danger_penalty = 0.0
    for seeker_coordinates in currently_visible_seekers_coordinates:
        distance_to_seeker = distance_metric(current_agent_coordinates, seeker_coordinates)
        if distance_to_seeker < visibility_radius:
            danger_penalty += danger_coefficient * (1 - distance_to_seeker / visibility_radius)
    return danger_penalty


def is_nonprogressive_move(*,
                           previous_time_steps_to_goal: Optional[int],
                           current_time_steps_to_goal: Optional[int]) -> bool:
    """
    Computes whether the agent's move was nonprogressive.
    
    :param previous_time_steps_to_goal: The previous minimum number of time steps to reach the goal before the agent's move.
    :param current_time_steps_to_goal: The current minimum number of time steps to reach the goal after the agent's move.
    """
    return (
        previous_time_steps_to_goal is not None
        and current_time_steps_to_goal is not None
        and current_time_steps_to_goal >= previous_time_steps_to_goal
    )


def is_nonprogress_penalizable(*,
                               previous_time_steps_to_goal: Optional[int],
                               current_time_steps_to_goal: Optional[int],
                               previously_visible_seekers_coordinates: Sequence[Coordinates]) -> bool:
    """
    Computes whether the agent's move should be penalized for not making progress towards the goal.
    
    :param previous_time_steps_to_goal: The previous minimum number of time steps to reach the goal before the agent's move.
    :param current_time_steps_to_goal: The current minimum number of time steps to reach the goal after the agent's move.
    :param previously_visible_seekers_coordinates: The coordinates of the seekers visible to the agent before the agent's move.

    :return: Whether the agent's move should be penalized for not making progress towards the goal.
    """
    return is_nonprogressive_move(previous_time_steps_to_goal = previous_time_steps_to_goal,
                                  current_time_steps_to_goal = current_time_steps_to_goal) and len(previously_visible_seekers_coordinates) == 0


def compute_nonprogress_penalty(*,
                                num_consecutive_penalizable_nonprogress_steps: int,
                                max_consecutive_penalizable_nonprogress_steps_to_penalize: int = 5,
                                base_nonprogress_penalty: float = 0.025,
                                nonprogress_coefficient: float = 0.005) -> float:
    """
    Computes the penalty term associated with the agent not making progress towards the goal.

    :param num_consecutive_penalizable_nonprogress_steps: The number of consecutive penalizable nonprogressive moves.
    :param max_consecutive_penalizable_nonprogress_steps_to_penalize: The maximum number of consecutive penalizable nonprogressive moves to penalize.
    :param base_nonprogress_penalty: The base penalty term associated with the agent not making progress towards the goal.
    :param nonprogress_coefficient: The coefficient for the nonprogress penalty term.
    :return: The penalty term associated with the agent not making progress towards the goal.
    """
    if num_consecutive_penalizable_nonprogress_steps == 0:
        return 0.0
    return base_nonprogress_penalty + nonprogress_coefficient * min(num_consecutive_penalizable_nonprogress_steps, max_consecutive_penalizable_nonprogress_steps_to_penalize)


def compute_rewards(*, 
                    goal: bool,
                    collided: bool,
                    intercepted: bool,
                    truncated: bool,
                    previous_time_steps_to_goal: Optional[int] = None,
                    current_time_steps_to_goal: Optional[int] = None,
                    progress_coefficient: float = 0.1,
                    current_agent_coordinates: Coordinates,
                    currently_visible_seekers_coordinates: Sequence[Coordinates],
                    visibility_radius: int,
                    distance_metric: Callable[[Coordinates, Coordinates], int | float] = chebyshev_distance,
                    danger_coefficient: float = 0.01,
                    num_consecutive_penalizable_nonprogress_steps: int,
                    max_consecutive_penalizable_nonprogress_steps_to_penalize: int = 5,
                    base_nonprogress_penalty: float = 0.025,
                    nonprogress_coefficient: float = 0.005,
                    step_penalty: float = 0.01) -> float:
    """
    Computes the agent's reward for performing an action in the environment.
    
    :param goal: Whether the agent reached the goal.
    :param collided: Whether the agent collided with an obstacle, a seeker, or an out-of-bounds cell.
    :param intercepted: Whether the agent was intercepted by a seeker.
    :param truncated: Whether the episode was truncated.
    :param previous_time_steps_to_goal: The previous minimum number of time steps to reach the goal.
    :param current_time_steps_to_goal: The current minimum number of time steps to reach the goal.
    :param progress_coefficient: The coefficient for the progress reward term.
    :param current_agent_coordinates: The current coordinates of the agent after the agent's move.
    :param currently_visible_seekers_coordinates: The coordinates of the seekers currently visible to the agent after the agent's move.
    :param visibility_radius: The visibility radius of the agent.
    :param distance_metric: The distance metric to use for computing the distance between the agent and a seeker.
    :param danger_coefficient: The coefficient for the danger penalty term.
    :param num_consecutive_penalizable_nonprogress_steps: The number of consecutive penalizable nonprogressive moves.
    :param max_consecutive_penalizable_nonprogress_steps_to_penalize: The maximum number of consecutive penalizable nonprogressive moves to penalize.
    :param base_nonprogress_penalty: The base penalty term associated with the agent not making progress towards the goal.
    :param nonprogress_coefficient: The coefficient for the nonprogress penalty term.
    :param step_penalty: The penalty term for each time step.
    :return: The agent's reward for performing an action in the environment.
    """
    terminal_rewards = compute_terminal_rewards(goal = goal,
                                                collided = collided,
                                                intercepted = intercepted,
                                                truncated = truncated)
    progress_rewards = compute_progress_rewards(previous_time_steps_to_goal = previous_time_steps_to_goal,
                                                current_time_steps_to_goal = current_time_steps_to_goal,
                                                progress_coefficient = progress_coefficient)
    danger_penalty = compute_danger_penalty(current_agent_coordinates = current_agent_coordinates,
                                            currently_visible_seekers_coordinates = currently_visible_seekers_coordinates,
                                            visibility_radius = visibility_radius,
                                            distance_metric = distance_metric,
                                            danger_coefficient = danger_coefficient)
    nonprogress_penalty = compute_nonprogress_penalty(num_consecutive_penalizable_nonprogress_steps = num_consecutive_penalizable_nonprogress_steps,
                                                      max_consecutive_penalizable_nonprogress_steps_to_penalize = max_consecutive_penalizable_nonprogress_steps_to_penalize,
                                                      base_nonprogress_penalty = base_nonprogress_penalty,
                                                      nonprogress_coefficient = nonprogress_coefficient)
    return terminal_rewards + progress_rewards - danger_penalty - nonprogress_penalty - step_penalty


def compute_visible_seekers(*,
                            current_agent_coordinates: Coordinates,
                            current_seekers_coordinates: Iterable[Coordinates],
                            visibility_radius: int,
                            visibility_rays: dict[Vector, list[Coordinates]],
                            obstacles_coordinates: AbstractSet[Coordinates]) -> list[Coordinates]:
    """
    Computes the coordinates of the seekers visible to the agent.
    """
    visible_seekers = []
    agent_offset = Vector(current_agent_coordinates.x, current_agent_coordinates.y)
    for seeker_coordinates in current_seekers_coordinates:
        if chebyshev_distance(current_agent_coordinates, seeker_coordinates) > visibility_radius:
            continue
        seeker_offset = seeker_coordinates - current_agent_coordinates
        visibility_ray = [coordinates + agent_offset for coordinates in visibility_rays[seeker_offset]]
        if not can_see(visibility_ray, obstacles_coordinates):
            continue
        visible_seekers.append(seeker_coordinates)
    return visible_seekers
