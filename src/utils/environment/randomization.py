"""
This module provides helper functions for randomizing the reactive avoidance environment.
"""
import numpy as np
from collections import deque
from typing import Optional
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.grid_dimensions import GridDimensions


def sample_free_cell(
    rng: np.random.Generator,
    grid_dimensions: GridDimensions,
    occupied_cells: set[Coordinates]
) -> Coordinates:
    """
    Samples a random unoccupied cell in the grid.
    
    :param rng: A numpy random number generator.
    :param grid_dimensions: The grid dimensions.
    :param occupied_cells: The set of occupied cells.
    :return: The coordinates of a random unoccupied cell.
    """
    while True:
        x = int(rng.integers(0, grid_dimensions.width))
        y = int(rng.integers(0, grid_dimensions.height))
        candidate = Coordinates(x, y)
        if candidate not in occupied_cells:
            return candidate


def sample_free_cell_with_min_distance(
    rng: np.random.Generator,
    grid_dimensions: GridDimensions,
    occupied_cells: set[Coordinates],
    reference: Coordinates,
    min_distance: int,
    max_attempts: int = 1000
) -> Coordinates:
    """
    Samples a random unoccupied cell at least min_distance (Chebyshev) away from a reference coordinate.
    
    :param rng: A numpy random number generator.
    :param grid_dimensions: The grid dimensions.
    :param occupied_cells: The set of occupied cells.
    :param reference: The reference coordinates.
    :param min_distance: The minimum Chebyshev distance from the reference.
    :param max_attempts: The maximum number of attempts before relaxing the constraint.
    :return: The coordinates of a random unoccupied cell satisfying the distance constraint.
    """
    for _ in range(max_attempts):
        candidate = sample_free_cell(rng, grid_dimensions, occupied_cells)
        if max(abs(candidate.x - reference.x), abs(candidate.y - reference.y)) >= min_distance:
            return candidate
    # Fallback: return any free cell if distance constraint can't be met
    return sample_free_cell(rng, grid_dimensions, occupied_cells)


def generate_random_obstacles(
    rng: np.random.Generator,
    grid_dimensions: GridDimensions,
    num_obstacles: int,
    reserved_cells: set[Coordinates],
    clump_probability: float = 0.3
) -> set[Coordinates]:
    """
    Generates random obstacle positions with optional clumping.
    
    Obstacles are placed one at a time. With probability clump_probability, a new obstacle
    is placed adjacent to an existing obstacle (creating clusters). Otherwise, it is placed
    at a random free cell.
    
    :param rng: A numpy random number generator.
    :param grid_dimensions: The grid dimensions.
    :param num_obstacles: The number of obstacles to place.
    :param reserved_cells: Cells that must not have obstacles (e.g., agent start, goal, seeker starts).
    :param clump_probability: The probability of placing an obstacle adjacent to an existing one.
    :return: A set of obstacle coordinates.
    """
    obstacles: set[Coordinates] = set()
    occupied = set(reserved_cells)

    for i in range(num_obstacles):
        placed = False

        # Try clumping: place adjacent to an existing obstacle
        if i > 0 and rng.random() < clump_probability:
            # Pick a random existing obstacle and try to place adjacent
            existing_list = list(obstacles)
            anchor = existing_list[rng.integers(0, len(existing_list))]
            neighbors = _get_adjacent_cells(anchor, grid_dimensions)
            rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in occupied:
                    obstacles.add(neighbor)
                    occupied.add(neighbor)
                    placed = True
                    break

        # Fallback: place at a random free cell
        if not placed:
            total_cells = grid_dimensions.width * grid_dimensions.height
            if len(occupied) >= total_cells:
                break
            cell = sample_free_cell(rng, grid_dimensions, occupied)
            obstacles.add(cell)
            occupied.add(cell)

    return obstacles


def _get_adjacent_cells(
    center: Coordinates,
    grid_dimensions: GridDimensions
) -> list[Coordinates]:
    """
    Returns the list of adjacent cells (4-connected) within the grid.
    
    :param center: The center coordinates.
    :param grid_dimensions: The grid dimensions.
    :return: A list of adjacent coordinates within the grid bounds.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    result = []
    for dx, dy in directions:
        neighbor = Coordinates(center.x + dx, center.y + dy)
        if grid_dimensions.contains_coordinates(neighbor):
            result.append(neighbor)
    return result


def has_path(
    start: Coordinates,
    goal: Coordinates,
    grid_dimensions: GridDimensions,
    obstacles: set[Coordinates]
) -> bool:
    """
    Checks if there is a path from start to goal using BFS (8-directional movement).
    
    :param start: The start coordinates.
    :param goal: The goal coordinates.
    :param grid_dimensions: The grid dimensions.
    :param obstacles: The obstacle coordinates.
    :return: True if a path exists, False otherwise.
    """
    if start == goal:
        return True
    if start in obstacles or goal in obstacles:
        return False

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    visited = {start}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for dx, dy in directions:
            neighbor = Coordinates(current.x + dx, current.y + dy)
            if neighbor == goal:
                return True
            if not grid_dimensions.contains_coordinates(neighbor):
                continue
            if neighbor in obstacles or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return False


def generate_randomized_layout(
    rng: np.random.Generator,
    grid_dimensions: GridDimensions,
    *,
    num_obstacles_range: tuple[int, int],
    num_seekers_range: tuple[int, int],
    seeker_policies: list[str],
    min_start_goal_distance: int,
    clump_probability: float = 0.3,
    max_retries: int = 50
) -> Optional[dict]:
    """
    Generates a full randomized layout (obstacles, agent start, goal, seekers) with
    path-reachability guarantee.
    
    :param rng: A numpy random number generator.
    :param grid_dimensions: The grid dimensions.
    :param num_obstacles_range: A tuple (min, max) for the number of obstacles.
    :param num_seekers_range: A tuple (min, max) for the number of seekers.
    :param seeker_policies: A list of seeker policies to sample from.
    :param min_start_goal_distance: The minimum Chebyshev distance between start and goal.
    :param clump_probability: The probability of clumping obstacles together.
    :param max_retries: The maximum number of retries to generate a valid layout.
    :return: A dictionary with keys 'obstacles', 'agent_start', 'goal', 'seekers' or None if generation fails.
    """
    for _ in range(max_retries):
        # 1. Pick agent start and goal first
        agent_start = sample_free_cell(rng, grid_dimensions, set())
        goal = sample_free_cell_with_min_distance(
            rng, grid_dimensions, {agent_start}, agent_start, min_start_goal_distance
        )

        # 2. Determine seeker count and positions
        num_seekers = int(rng.integers(num_seekers_range[0], num_seekers_range[1] + 1))
        reserved = {agent_start, goal}
        seeker_positions: list[Coordinates] = []
        for _ in range(num_seekers):
            seeker_pos = sample_free_cell(rng, grid_dimensions, reserved)
            seeker_positions.append(seeker_pos)
            reserved.add(seeker_pos)

        # 3. Generate obstacles (avoiding reserved cells)
        num_obstacles = int(rng.integers(num_obstacles_range[0], num_obstacles_range[1] + 1))
        obstacles = generate_random_obstacles(
            rng, grid_dimensions, num_obstacles, reserved, clump_probability
        )

        # 4. Verify path exists from agent start to goal
        if not has_path(agent_start, goal, grid_dimensions, obstacles):
            continue

        # 5. Assign random policies to seekers
        seekers = []
        for pos in seeker_positions:
            policy = seeker_policies[int(rng.integers(0, len(seeker_policies)))]
            seekers.append({
                "start_coordinates": pos,
                "velocity": 1,
                "policy": policy
            })

        return {
            "obstacles": obstacles,
            "agent_start": agent_start,
            "goal": goal,
            "seekers": seekers
        }

    return None
