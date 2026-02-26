"""
This module provides the ReactiveAvoidanceEnvironment class for representing a two-dimensional gridworld environment where an agent traverses from some start coordinates to some goal coordinates while avoiding static obstacles and one or more seekers.
"""
import numpy as np
from collections import deque
from pathlib import Path
from typing import Any, cast, Optional
from gymnasium.spaces import Discrete
from minigrid.core.grid import Grid  
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from stable_baselines3.common.env_checker import check_env
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.helpers import calculate_observation_space, scale_absolute_coordinates, scale_relative_vector
from src.utils.environment.seeker import Seeker
from src.utils.environment.vector import Vector
from src.utils.yaml_parser.configuration import SystemConfiguration


class ReactiveAvoidanceEnv(MiniGridEnv):
    """
    A class for representing a two-dimensional gridworld environment where an agent traverses from some start coordinates to some goal coordinates while avoiding static obstacles and one or more seekers.
    """
    def __init__(self, config: SystemConfiguration, render_mode: Optional[str] = None) -> None:
        """
        Creates a ReactiveAvoidanceEnv object.
        
        :param config: A SystemConfiguration object representing the configuration of the agent-seeker-gridworld system.
        :param render_mode: The rendering mode for the environment.
        :return: None.
        """
        super().__init__(
            mission_space = MissionSpace(mission_func = lambda: "Traverse from some start coordinates to some goal coordinates while avoiding static obstacles and one or more seekers."),
            width = config.environment.grid_dimensions.width,
            height = config.environment.grid_dimensions.height,
            max_steps = config.environment.episode_time_limit,
            # TODO: Set see_through_walls to False and handle occlusion
            see_through_walls = True,
            render_mode = render_mode
        )
        self.config = config
        # Initialize the internal dynamic components of the environment state
        self._current_step = 0
        self._current_agent_coordinates = config.agent.start_coordinates
        self._current_seeker_coordinates = [seeker.start_coordinates for seeker in config.seekers]
        current_local_observation = self._compute_local_observation()
        self._current_local_observation_history: deque[np.ndarray] = deque(maxlen = config.agent.observation_stack_depth)
        for _ in range(config.agent.observation_stack_depth):
            self._current_local_observation_history.append(current_local_observation)

        # Override the default action space and observation space
        self.action_space = Discrete(9)
        self._action_map = {
            0: Vector(0, 0),   # No-op
            1: Vector(0, 1),  # Up
            2: Vector(1, 0),   # Right
            3: Vector(0, -1),   # Down
            4: Vector(-1, 0),  # Left
            5: Vector(1, 1),  # Up-Right
            6: Vector(1, -1),   # Down-Right
            7: Vector(-1, 1), # Up-Left
            8: Vector(-1, -1),  # Down-Left
        }
        self.observation_space = calculate_observation_space(config.agent.visibility_radius, config.agent.observation_stack_depth)

    def _compute_local_observation(self) -> np.ndarray:
        """
        Computes the agent's local observation for the current environment state. For each cell within the agent's visibility radius, the local observation includes whether the agent observes an obstacle, whether the agent observes a seeker, and whether the agent observes the goal.
        
        :return: The local observation for the current environment state.
        """
        visibility_radius = self.config.agent.visibility_radius
        visibility_diameter = 2 * visibility_radius + 1
        # TODO: Add two channels: whether the agent observes an out-of-bounds cell and whether the agent observes an occluded cell
        # TODO: If the agent observes an occluded cell, all other channels should be set to 0
        local_observation = np.zeros((3, visibility_diameter, visibility_diameter), dtype = np.float32)
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                current_coordinates = Coordinates(self._current_agent_coordinates.x + dx, self._current_agent_coordinates.y + dy)
                if not self.config.environment.grid_dimensions.contains_coordinates(current_coordinates):
                    continue
                if current_coordinates in self.config.environment.obstacles_coordinates:
                    local_observation[0, dx + visibility_radius, dy + visibility_radius] = 1.0
                if current_coordinates in self._current_seeker_coordinates:
                    local_observation[1, dx + visibility_radius, dy + visibility_radius] = 1.0
                if current_coordinates == self.config.agent.goal_coordinates:
                    local_observation[2, dx + visibility_radius, dy + visibility_radius] = 1.0
        return local_observation.reshape(-1)

    def _get_obs(self) -> np.ndarray:
        """
        Converts the environment state into an observation vector.
        
        :return: The observation vector.
        """
        scaled_current_agent_coordinates = scale_absolute_coordinates(self._current_agent_coordinates,
                                                                      self.config.environment.grid_dimensions.width, self.config.environment.grid_dimensions.height)
        scaled_relative_goal_vector = scale_relative_vector(self.config.agent.goal_coordinates - self._current_agent_coordinates,
                                                            self.config.environment.grid_dimensions.width, self.config.environment.grid_dimensions.height)
        observation_stack = np.concatenate(self._current_local_observation_history, dtype = np.float32)
        return np.concatenate((scaled_current_agent_coordinates, scaled_relative_goal_vector, observation_stack), dtype = np.float32)

    def _get_info(self, *, outcome: Optional[str] = None) -> dict[str, Any]:
        """
        Computes auxiliary diagnostic information about the current environment state and the current episode.
        
        :param outcome: The outcome of the current episode.
        :return: A dictionary of auxiliary diagnostic information.
        """
        return {
            "current_step": self._current_step,
            "current_agent_coordinates": self._current_agent_coordinates,
            "current_seeker_coordinates": self._current_seeker_coordinates,
            "current_local_observation_history": self._current_local_observation_history,
            "outcome": outcome
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Resets the environment to its initial state.
        
        :param seed: The seed for the environment's random number generator.
        :param options: A dictionary of options specifying how to reset the environment.
        """
        # Calls self._gen_grid() internally to set up the grid for the reactive avoidance environment
        super().reset(seed = seed, options = options)
        self._current_step = 0
        self._current_agent_coordinates = self.config.agent.start_coordinates
        self._current_seeker_coordinates = [seeker.start_coordinates for seeker in self.config.seekers]
        current_local_observation = self._compute_local_observation()
        self._current_local_observation_history = deque(maxlen = self.config.agent.observation_stack_depth)
        for _ in range(self.config.agent.observation_stack_depth):
            self._current_local_observation_history.append(current_local_observation)
        return self._get_obs(), self._get_info() 

    def _gen_grid(self, width: int, height: int) -> None:
        """
        Implements the _gen_grid abstract method from the MiniGridEnv class. Sets up the grid for the reactive avoidance environment.
        
        :param width: The width of the grid.
        :param height: The height of the grid.
        :return: None.
        """
        # Initialize the grid
        self.grid = Grid(width, height)

        # Place agent on the grid
        self.agent_pos = (self.config.agent.start_coordinates.x, self.config.agent.start_coordinates.y)
        # Set the agent's direction in the grid 
        # NOTE: Required for the MiniGridEnv class
        self.agent_dir = 0

        # Place goal on the grid
        self.grid.set(self.config.agent.goal_coordinates.x, self.config.agent.goal_coordinates.y, Goal())
    
        # Place seekers on the grid
        for seeker in self.config.seekers:
            self.grid.set(seeker.start_coordinates.x, seeker.start_coordinates.y, Seeker())
        
        # Place obstacles on the grid
        for obstacle in self.config.environment.obstacles_coordinates:
            self.grid.set(obstacle.x, obstacle.y, Wall())

    def _can_agent_be_in(self, coordinates: Coordinates) -> bool:
        """
        Checks if the agent can be in the given coordinates.
        
        :param coordinates: The coordinates to check.
        :return: True if the agent can be in the given coordinates, False otherwise.
        """
        if not self.config.environment.grid_dimensions.contains_coordinates(coordinates):
            return False
        if coordinates in self.config.environment.obstacles_coordinates:
            return False
        if coordinates in self._current_seeker_coordinates:
            return False
        return True

    def _move_agent(self, action: int) -> Coordinates:
        """
        Moves the agent in the environment based on the given action.
        
        :param action: The action to take in the environment.
        :return: The new coordinates of the agent.
        """
        movement_vector = self._action_map[action]
        for _ in range(self.config.agent.velocity):
            # Move the agent in the selected direction
            # NOTE: This movement allows the agent to 'slip' through diagonal obstacles
            # TODO: Create shared motion logic between environment and validator
            self._current_agent_coordinates += movement_vector
            # If the agent cannot be in the new coordinates or if the agent has reached the goal, return the new coordinates
            if not self._can_agent_be_in(self._current_agent_coordinates) or self._current_agent_coordinates == self.config.agent.goal_coordinates:
                # Update the agent's coordinates in the grid
                self.agent_pos = (self._current_agent_coordinates.x, self._current_agent_coordinates.y)
                return self._current_agent_coordinates
        # Update the agent's coordinates in the grid
        self.agent_pos = (self._current_agent_coordinates.x, self._current_agent_coordinates.y)
        # Return the final coordinates
        return self._current_agent_coordinates

    def _seeker_policy(self, current_coordinates: Coordinates, current_agent_coordinates: Coordinates) -> Vector:
        """
        Returns the movement vector that minimizes the Chebyshev distance between the seeker and the agent.
        
        :param current_coordinates: The current coordinates of the seeker.
        :param current_agent_coordinates: The current coordinates of the agent.
        :return: The movement vector that minimizes the Chebyshev distance between the seeker and the agent.
        """
        return Vector(
            np.sign(current_agent_coordinates.x - current_coordinates.x).astype(int),
            np.sign(current_agent_coordinates.y - current_coordinates.y).astype(int)
        )

    def _move_seeker(self, current_coordinates: Coordinates, velocity: int) -> Coordinates:
        """
        Moves the seeker in the environment based on its current coordinates and velocity.  The seeker picks the action that minimizes Chebyshev distance to the agent.
        
        :param current_coordinates: The current coordinates of the seeker.
        :param velocity: The velocity of the seeker.
        :return: The new coordinates of the seeker.
        """
        # Remove the old seeker from the grid
        self.grid.set(current_coordinates.x, current_coordinates.y, None)
        for _ in range(velocity):
            movement_vector = self._seeker_policy(current_coordinates, self._current_agent_coordinates)
            new_coordinates = current_coordinates + movement_vector
            # If the seeker cannot be in the new coordinates or if the new coordinates are the goal coordinates, return the old coordinates
            if not self._can_agent_be_in(new_coordinates) or new_coordinates == self.config.agent.goal_coordinates:
                # Update the seeker's coordinates in the grid
                self.grid.set(current_coordinates.x, current_coordinates.y, Seeker())
                return current_coordinates
            # Otherwise, move the seeker in the selected direction
            current_coordinates += movement_vector
        # Update the seeker's coordinates in the grid
        self.grid.set(current_coordinates.x, current_coordinates.y, Seeker())
        return current_coordinates

    def _finalize_step(self, *, reward: float, terminated: bool, truncated: bool, outcome: Optional[str]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Finalizes the step in the environment. Computes and appends the agent's local observation to the local observation history. Returns a tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        
        :param reward: The agent's reward for the step.
        :param terminated: The termination flag for the episode.
        :param truncated: The truncation flag for the episode.
        :param outcome: The outcome of the episode.
        :return: A tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        """
        self._current_local_observation_history.append(self._compute_local_observation())
        return self._get_obs(), reward, terminated, truncated, self._get_info(outcome = outcome)

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Performs a single step in the environment.
        
        :param action: The action to take in the environment.
        :return: A tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        """
        # TODO: Reward shaping
        action = cast(int, action)
        self._current_step += 1

        # Move the agent
        self._current_agent_coordinates = self._move_agent(action)

        truncated = self._current_step >= self.config.environment.episode_time_limit

        # If moving the agent results in a collision, return the current observation and terminate the episode
        if not self._can_agent_be_in(self._current_agent_coordinates):
            return self._finalize_step(reward = -1.0, terminated = True, truncated = truncated, outcome = "collision")

        # If moving the agent results in reaching the goal, return the current observation and terminate the episode
        if self._current_agent_coordinates == self.config.agent.goal_coordinates:
            return self._finalize_step(reward = 1.0, terminated = True, truncated = truncated, outcome = "goal")

        # Move the seekers
        for i, seeker in enumerate(self.config.seekers):
            self._current_seeker_coordinates[i] = self._move_seeker(self._current_seeker_coordinates[i], seeker.velocity)

        # If moving the seekers results in an interception, return the current observation and terminate the episode
        if self._current_agent_coordinates in self._current_seeker_coordinates:
            return self._finalize_step(reward = -1.0, terminated = True, truncated = truncated, outcome = "interception")

        # Otherwise, continue the episode
        return self._finalize_step(reward = -1 / self.config.environment.episode_time_limit, terminated = False, truncated = truncated, outcome = "timeout" if truncated else None)

    # def step(self, action):

    #     if isinstance(action, np.ndarray):
    #         action = int(action.item())

    #     # Get movement direction
    #     direction = self.action_map[action]

    #     # Calculate new position
    #     new_pos = (
    #         self.agent_pos[0] + direction[0],
    #         self.agent_pos[1] + direction[1]
    #     )

    #     # Clip to grid boundaries
    #     new_pos = (
    #         int(np.clip(new_pos[0], 0, self.width - 1)),
    #         int(np.clip(new_pos[1], 0, self.height - 1))
    #     )

    #     prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))

    #     # Check if cell is passable (seekers have can_overlap=True, so agent can step on them)
    #     cell = self.grid.get(*new_pos)
    #     if cell is None or cell.can_overlap():
    #         self.agent_pos = new_pos
        
    #     # Increment step counter
    #     self.step_count += 1

    #     # Move seekers toward the agent after the agent moves
    #     self._move_seekers()

    #     # Check if a seeker caught the agent
    #     if self._seeker_caught_agent():
    #         observation = self.gen_obs()
    #         return observation, -100.0, True, False, {}

    #     # Check if reached goal
    #     if tuple(self.agent_pos) == self.goal_pos:
    #         reward = 100.0
    #         terminated = True
    #     else:
    #         curr_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
    #         dist_reward = (prev_dist - curr_dist) * 0.5

    #         # Proximity bonus using inverse distance so it spikes sharply when
    #         # adjacent to the goal (~8.0 at dist=1), overwhelming any seeker-fear
    #         # and preventing oscillation. Falls to ~0 beyond 5 cells.
    #         if curr_dist < 5.0:
    #             proximity_bonus = min(8.0, 8.0 / max(curr_dist, 0.5)) - 1.6
    #         else:
    #             proximity_bonus = 0.0

    #         reward = -0.1 + dist_reward + proximity_bonus
    #         terminated = False

    #     # Check if max steps reached
    #     truncated = self.step_count >= self.max_steps

    #     observation = self.gen_obs()
    #     return observation, reward, terminated, truncated, {}

    def get_full_render(self, highlight: bool = True, tile_size: Optional[int] = None) -> np.ndarray:
        """
        Overrides the get_full_render method from the MiniGridEnv class. Renders the full environment as an RGB image. Highlights a square of radius self.config.agent.visibility_radius around the agent.
        
        :param highlight: Whether to highlight the square of radius self.config.agent.visibility_radius around the agent.
        :param tile_size: The size of each tile in the rendered image.
        :return: The rendered image as an RGB array.
        """
        if tile_size is None:
            tile_size = self.tile_size

        highlight_mask = np.zeros((self.config.environment.grid_dimensions.width, self.config.environment.grid_dimensions.height), dtype = bool)

        if highlight:
            visibility_radius = self.config.agent.visibility_radius
            for dx in range(-visibility_radius, visibility_radius + 1):
                for dy in range(-visibility_radius, visibility_radius + 1):
                    current_coordinates = Coordinates(self._current_agent_coordinates.x + dx, self._current_agent_coordinates.y + dy)
                    if not self.config.environment.grid_dimensions.contains_coordinates(current_coordinates):
                        continue
                    highlight_mask[current_coordinates.x, current_coordinates.y] = True

        return self.grid.render(
            tile_size,
            (self._current_agent_coordinates.x, self._current_agent_coordinates.y),
            self.agent_dir,
            highlight_mask = highlight_mask
        )


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
