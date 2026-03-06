"""
This module provides the ReactiveAvoidanceEnvironment class for representing a two-dimensional gridworld environment where an agent traverses from some start coordinates to some goal coordinates while avoiding static obstacles and one or more seekers.
"""
import numpy as np
from collections import deque
from typing import Any, cast, Literal, Optional, Union
from gymnasium.spaces import Discrete
from minigrid.core.grid import Grid  
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.helpers import (
    calculate_observation_space,
    can_see,
    chebyshev_distance,
    compute_offset_to_line_mapping,
    compute_time_steps_to_goal_mapping,
    scale_absolute_coordinates,
    scale_relative_vector,
    move_agent,
    move_seeker
)
from src.utils.environment.seeker import Seeker
from src.utils.environment.vector import Vector
from src.utils.logging.helpers import compute_episode_metrics, compute_distance_to_hazards
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
            see_through_walls = False,
            render_mode = render_mode
        )
        self.config = config
        # Compute caches
        self._visibility_rays = compute_offset_to_line_mapping(config.agent.visibility_radius)
        self._time_steps_to_goal_mapping = compute_time_steps_to_goal_mapping(
            goal_coordinates = config.agent.goal_coordinates,
            grid_dimensions = config.environment.grid_dimensions,
            obstacles_coordinates = config.environment.obstacles_coordinates,
            agent_velocity = config.agent.velocity
        )

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

        # Compute metrics
        self._minimum_distance_to_hazards = compute_distance_to_hazards(
            current_agent_coordinates = config.agent.start_coordinates,
            obstacles_coordinates = config.environment.obstacles_coordinates,
            grid_dimensions = config.environment.grid_dimensions,
            seekers_coordinates = self._current_seeker_coordinates
        )

    def _compute_local_observation(self) -> np.ndarray:
        """
        Computes the agent's local observation for the current environment state. For each cell within the agent's visibility radius, the local observation includes whether the agent observes an obstacle, whether the agent observes a seeker, and whether the agent observes the goal.
        
        :return: The local observation for the current environment state.
        """
        visibility_radius = self.config.agent.visibility_radius
        visibility_diameter = 2 * visibility_radius + 1

        seeker_coordinates = set(self._current_seeker_coordinates)

        local_observation = np.zeros((4, visibility_diameter, visibility_diameter), dtype = np.float32)
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                current_visibility_offset = Vector(dx, dy)
                current_cell_coordinates = self._current_agent_coordinates + current_visibility_offset
                current_agent_offset = Vector(self._current_agent_coordinates.x, self._current_agent_coordinates.y)
                current_visibility_ray = [coordinates + current_agent_offset for coordinates in self._visibility_rays[current_visibility_offset]]
                if not can_see(current_visibility_ray, self.config.environment.obstacles_coordinates):
                    local_observation[0, dx + visibility_radius, dy + visibility_radius] = 1.0
                    continue
                if not self.config.environment.grid_dimensions.contains_coordinates(current_cell_coordinates):
                    local_observation[1, dx + visibility_radius, dy + visibility_radius] = 1.0
                    continue
                if current_cell_coordinates in self.config.environment.obstacles_coordinates:
                    local_observation[2, dx + visibility_radius, dy + visibility_radius] = 1.0
                if current_cell_coordinates in seeker_coordinates:
                    local_observation[3, dx + visibility_radius, dy + visibility_radius] = 1.0
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

    def _get_info(self, *, episode_metrics: Optional[dict[str, Optional[Union[int, float, str]]]] = None) -> dict[str, Any]:
        """
        Computes auxiliary diagnostic information about the current environment state and the current episode.
        
        :param outcome: The outcome of the current episode.
        :return: A dictionary of auxiliary diagnostic information.
        """
        environment_state = {
            "current_step": self._current_step,
            "current_agent_coordinates": self._current_agent_coordinates,
            "current_seeker_coordinates": self._current_seeker_coordinates,
            "current_local_observation_history": self._current_local_observation_history,
        }
        if episode_metrics is None:
            return environment_state
        return environment_state | episode_metrics

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
        self._minimum_distance_to_hazards = compute_distance_to_hazards(
            current_agent_coordinates = self._current_agent_coordinates,
            obstacles_coordinates = self.config.environment.obstacles_coordinates,
            grid_dimensions = self.config.environment.grid_dimensions,
            seekers_coordinates = self._current_seeker_coordinates
        )
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

    def _finalize_step(self, *, reward: float, terminated: bool, truncated: bool,
                       outcome: Optional[str],
                       starting_time_steps_to_goal: Optional[int],
                       ending_time_steps_to_goal: Optional[int],
                       episode_length: int,
                       minimum_distance_to_hazards: dict[Literal["obstacle", "boundary", "seeker"], int],
                       collision_type: Optional[str]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Finalizes the step in the environment. Computes and appends the agent's local observation to the local observation history. Computes the episode metrics.
        Returns a tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        
        :param reward: The agent's reward for the step.
        :param terminated: The termination flag for the episode.
        :param truncated: The truncation flag for the episode.
        :param outcome: The outcome of the episode.
        :param starting_time_steps_to_goal: The minimum number of time steps to the goal at the start of the episode.
        :param ending_time_steps_to_goal: The minimum number of time steps to the goal at the end of the episode.
        :param episode_length: The length of the episode.
        :param minimum_distance_to_hazards: A dictionary mapping the type of hazard to the agent's minimum Chebyshev distance to the hazard.
        :param collision_type: The type of collision that occurred during the episode.
        :return: A tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        """
        self._current_local_observation_history.append(self._compute_local_observation())
        episode_metrics = compute_episode_metrics(outcome = outcome,
                                                  starting_time_steps_to_goal = starting_time_steps_to_goal,
                                                  ending_time_steps_to_goal = ending_time_steps_to_goal,
                                                  episode_length = episode_length,
                                                  minimum_distance_to_hazards = minimum_distance_to_hazards,
                                                  collision_type = collision_type)
        return self._get_obs(), reward, terminated, truncated, self._get_info(episode_metrics = episode_metrics)

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Performs a single step in the environment.
        
        :param action: The action to take in the environment.
        :return: A tuple containing the observation, reward, termination flag, truncation flag, and auxiliary info dictionary.
        """
        action = cast(int, action)
        self._current_step += 1
        truncated = self._current_step >= self.config.environment.episode_time_limit

        # Compute time steps to goal before moving
        previous_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates)

        # Move the agent in the environment
        self._current_agent_coordinates, collided, goal = move_agent(current_coordinates = self._current_agent_coordinates,
                                                                     velocity = self.config.agent.velocity,
                                                                     movement_vector = self._action_map[action],
                                                                     grid_dimensions = self.config.environment.grid_dimensions,
                                                                     obstacles_coordinates = self.config.environment.obstacles_coordinates,
                                                                     seekers_coordinates = self._current_seeker_coordinates,
                                                                     goal_coordinates = self.config.agent.goal_coordinates)
        # Update the agent's position in the grid
        self.agent_pos = (self._current_agent_coordinates.x, self._current_agent_coordinates.y)

        # Compute minimum distance to hazards after moving
        current_distance_to_hazards = compute_distance_to_hazards(current_agent_coordinates = self._current_agent_coordinates,
                                                                  obstacles_coordinates = self.config.environment.obstacles_coordinates,
                                                                  grid_dimensions = self.config.environment.grid_dimensions,
                                                                  seekers_coordinates = self._current_seeker_coordinates)
        self._minimum_distance_to_hazards: dict[Literal["obstacle", "boundary", "seeker"], int] = {key: min(self._minimum_distance_to_hazards[key], value) for key, value in current_distance_to_hazards.items()}

        # If moving the agent results in a collision, return the current observation and terminate the episode
        if collided:
            collision_type = None
            if self._current_agent_coordinates in self.config.environment.obstacles_coordinates:
                collision_type = "obstacle"
            elif not self.config.environment.grid_dimensions.contains_coordinates(self._current_agent_coordinates):
                collision_type = "boundary"
            elif self._current_agent_coordinates in self._current_seeker_coordinates:
                collision_type = "seeker"
            return self._finalize_step(reward = -5, terminated = True, truncated = truncated,
                                       outcome = "collision",
                                       starting_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self.config.agent.start_coordinates),
                                       ending_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates),
                                       episode_length = self._current_step,
                                       minimum_distance_to_hazards = self._minimum_distance_to_hazards,
                                       collision_type = collision_type)
        # If moving the agent results in a goal, return the current observation and terminate the episode
        if goal:
            return self._finalize_step(reward = 5, terminated = True, truncated = truncated,
                                       outcome = "goal",
                                       starting_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self.config.agent.start_coordinates),
                                       ending_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates),
                                       episode_length = self._current_step,
                                       minimum_distance_to_hazards = self._minimum_distance_to_hazards,
                                       collision_type = None)

        # Clear the seekers' positions in the grid
        for coordinates in self._current_seeker_coordinates:
            self.grid.set(coordinates.x, coordinates.y, None)
        # Move the seekers in the environment
        for i, seeker in enumerate(self.config.seekers):
            other_seekers_coordinates = self._current_seeker_coordinates[:i] + self._current_seeker_coordinates[i+1:]
            # TODO: Implement other policies
            self._current_seeker_coordinates[i] = move_seeker(current_coordinates = self._current_seeker_coordinates[i],
                                                              velocity = seeker.velocity,
                                                              policy = "greedy",
                                                              current_agent_coordinates = self._current_agent_coordinates,
                                                              grid_dimensions = self.config.environment.grid_dimensions,
                                                              obstacles_coordinates = self.config.environment.obstacles_coordinates,
                                                              seekers_coordinates = other_seekers_coordinates,
                                                              goal_coordinates = self.config.agent.goal_coordinates)
        # Update the seekers' positions in the grid
        for coordinates in self._current_seeker_coordinates:
            self.grid.set(coordinates.x, coordinates.y, Seeker())

        # Compute minimum distance to hazards after moving the seekers
        current_distance_to_hazards = compute_distance_to_hazards(current_agent_coordinates = self._current_agent_coordinates,
                                                                  obstacles_coordinates = self.config.environment.obstacles_coordinates,
                                                                  grid_dimensions = self.config.environment.grid_dimensions,
                                                                  seekers_coordinates = self._current_seeker_coordinates)
        self._minimum_distance_to_hazards = {key: min(self._minimum_distance_to_hazards[key], value) for key, value in current_distance_to_hazards.items()}

        # If moving the seekers results in an interception, return the current observation and terminate the episode
        if self._current_agent_coordinates in self._current_seeker_coordinates:
            return self._finalize_step(reward = -5, terminated = True, truncated = truncated,
                                       outcome = "interception",
                                       starting_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self.config.agent.start_coordinates),
                                       ending_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates),
                                       episode_length = self._current_step,
                                       minimum_distance_to_hazards = self._minimum_distance_to_hazards,
                                       collision_type = None)

        # Compute time steps to goal after moving
        current_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates)

        # Compute progress reward
        if previous_time_steps_to_goal is None or current_time_steps_to_goal is None:
            progress_reward = 0.0
        else:
            # TODO: Implement coefficient for progress reward
            progress_reward = (previous_time_steps_to_goal - current_time_steps_to_goal) * 0.05
        # Linear danger penalty within a Chebyshevradius of 3 cells
        danger_penalty = 0.0
        for seeker_coordinates in self._current_seeker_coordinates:
            distance_to_seeker = chebyshev_distance(self._current_agent_coordinates, seeker_coordinates)
            if distance_to_seeker < 3:
                danger_penalty -= 0.1 * (3 - distance_to_seeker) / 3
        # Step penalty to encourage efficiency
        step_penalty = -1 / self.config.environment.episode_time_limit

        # Otherwise, continue the episode
        return self._finalize_step(reward = step_penalty + progress_reward + danger_penalty, terminated = False, truncated = truncated,
                                   outcome = "timeout" if truncated else None,
                                   starting_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self.config.agent.start_coordinates),
                                   ending_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._current_agent_coordinates),
                                   episode_length = self._current_step,
                                   minimum_distance_to_hazards = self._minimum_distance_to_hazards,
                                   collision_type = None)

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
