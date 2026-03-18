"""
This module provides the MultiAgentReactiveAvoidanceEnv class, a PettingZoo-compatible
parallel environment where multiple runner agents navigate to a shared goal while
avoiding seekers and obstacles.
"""
import functools
import numpy as np
from collections import deque
from typing import Any, Optional, Literal

from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

from src.utils.environment.coordinates import Coordinates
from src.utils.environment.helpers import (
    calculate_observation_space,
    can_see,
    compute_offset_to_line_mapping,
    compute_rewards,
    compute_time_steps_to_goal_mapping,
    compute_visible_seekers,
    is_nonprogress_penalizable,
    scale_absolute_coordinates,
    scale_relative_vector,
    move_agent,
    move_seeker,
    MOVEMENT_VECTORS
)
from src.utils.environment.randomization import generate_randomized_layout
from src.utils.environment.seeker import Seeker
from src.utils.environment.vector import Vector
from src.utils.logging.helpers import (
    compute_distance_to_hazards,
    compute_minimum_distance_to_hazards
)
from src.utils.yaml_parser.configuration import SeekerConfiguration, SystemConfiguration


class MultiAgentReactiveAvoidanceEnv(ParallelEnv):
    """
    A PettingZoo-compatible parallel environment where multiple runner agents
    navigate a shared gridworld to reach a goal while avoiding seekers and obstacles.
    
    All runners share the same grid, obstacles, seekers, and goal. Each runner
    has its own observation (local visibility) and acts independently.
    
    Episodes end when ALL runners have either reached the goal, been intercepted,
    collided, or the time limit is reached.
    """

    metadata = {"render_modes": ["rgb_array"], "name": "multi_agent_reactive_avoidance_v0"}

    def __init__(self, config: SystemConfiguration, num_runners: int = 2, render_mode: Optional[str] = None):
        """
        Creates a MultiAgentReactiveAvoidanceEnv.
        
        :param config: A SystemConfiguration object.
        :param num_runners: Number of runner agents.
        :param render_mode: The rendering mode.
        """
        super().__init__()
        self.config = config
        self.num_runners = num_runners
        self.render_mode = render_mode
        self._randomization = config.randomization

        # Agent IDs
        self.possible_agents = [f"runner_{i}" for i in range(num_runners)]
        self.agents = list(self.possible_agents)

        # Runtime state (may be randomized on reset)
        self._runtime_obstacles: set[Coordinates] = set(config.environment.obstacles_coordinates)
        self._runtime_goal: Coordinates = config.agent.goal_coordinates
        self._runtime_seekers: list[SeekerConfiguration] = list(config.seekers)

        # Caches
        self._visibility_rays = compute_offset_to_line_mapping(config.agent.visibility_radius)
        self._time_steps_to_goal_mapping: dict[Coordinates, int] = {}

        # Per-agent state
        self._runner_coordinates: dict[str, Coordinates] = {}
        self._runner_observation_histories: dict[str, deque] = {}
        self._runner_active: dict[str, bool] = {}
        self._runner_nonprogress_steps: dict[str, int] = {}
        self._runner_min_distance_to_hazards: dict[str, dict] = {}
        self._runner_starting_time_steps: dict[str, Optional[int]] = {}

        # Shared state
        self._current_seekers_coordinates: list[Coordinates] = []
        self._current_step = 0
        self._starting_time_steps_to_goal: Optional[int] = None

        # Spaces
        self._obs_space = calculate_observation_space(config.agent.visibility_radius, config.agent.observation_stack_depth)
        self._action_space = Discrete(len(MOVEMENT_VECTORS))
        self._action_map = {index: mv for index, mv in enumerate(MOVEMENT_VECTORS)}

        # RNG
        self._rng = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Box:
        """Returns the observation space for the given agent."""
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Discrete:
        """Returns the action space for the given agent."""
        return self._action_space

    def _randomize_layout(self) -> None:
        """Randomizes the environment layout if configured."""
        if self._randomization is None or not self._randomization.is_enabled:
            return

        layout = generate_randomized_layout(
            rng = self._rng,
            grid_dimensions = self.config.environment.grid_dimensions,
            num_obstacles_range = self._randomization.num_obstacles_range,
            num_seekers_range = self._randomization.num_seekers_range,
            seeker_policies = self._randomization.seeker_policies,
            min_start_goal_distance = self._randomization.min_start_goal_distance
        )

        if layout is None:
            return

        if self._randomization.randomize_obstacles:
            self._runtime_obstacles = layout["obstacles"]

        if self._randomization.randomize_positions:
            self._runtime_goal = layout["goal"]
            self._runtime_seekers = [
                SeekerConfiguration(
                    start_coordinates = s["start_coordinates"],
                    velocity = s["velocity"],
                    policy = s["policy"]
                )
                for s in layout["seekers"]
            ]

    def _compute_local_observation(self, agent_coordinates: Coordinates) -> np.ndarray:
        """Computes the local observation for a runner at given coordinates."""
        visibility_radius = self.config.agent.visibility_radius
        visibility_diameter = 2 * visibility_radius + 1
        seekers_set = set(self._current_seekers_coordinates)

        # Include other runners' positions as "seekers" in observation (entities to avoid)
        other_runners = set()
        for agent_id, coords in self._runner_coordinates.items():
            if coords != agent_coordinates and self._runner_active.get(agent_id, False):
                other_runners.add(coords)

        local_observation = np.zeros((4, visibility_diameter, visibility_diameter), dtype=np.float32)
        for dx in range(-visibility_radius, visibility_radius + 1):
            for dy in range(-visibility_radius, visibility_radius + 1):
                offset = Vector(dx, dy)
                agent_offset = Vector(agent_coordinates.x, agent_coordinates.y)
                ray = [c + agent_offset for c in self._visibility_rays[offset]]
                if not can_see(ray, self._runtime_obstacles):
                    local_observation[0, dx + visibility_radius, dy + visibility_radius] = 1.0
                    continue
                cell = agent_coordinates + offset
                if not self.config.environment.grid_dimensions.contains_coordinates(cell):
                    local_observation[1, dx + visibility_radius, dy + visibility_radius] = 1.0
                    continue
                if cell in self._runtime_obstacles:
                    local_observation[2, dx + visibility_radius, dy + visibility_radius] = 1.0
                if cell in seekers_set or cell in other_runners:
                    local_observation[3, dx + visibility_radius, dy + visibility_radius] = 1.0
        return local_observation.reshape(-1)

    def _get_obs(self, agent: str) -> np.ndarray:
        """Gets the observation for a specific runner agent."""
        coords = self._runner_coordinates[agent]
        scaled_coords = scale_absolute_coordinates(
            coords, self.config.environment.grid_dimensions.width, self.config.environment.grid_dimensions.height
        )
        scaled_goal = scale_relative_vector(
            self._runtime_goal - coords,
            self.config.environment.grid_dimensions.width, self.config.environment.grid_dimensions.height
        )
        obs_stack = np.concatenate(self._runner_observation_histories[agent], dtype=np.float32)
        return np.concatenate((scaled_coords, scaled_goal, obs_stack), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """
        Resets the environment and all runner agents.
        
        :param seed: Random seed.
        :param options: Reset options.
        :return: Tuple of (observations dict, infos dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)

        # Reset runtime state
        self._runtime_obstacles = set(self.config.environment.obstacles_coordinates)
        self._runtime_goal = self.config.agent.goal_coordinates
        self._runtime_seekers = list(self.config.seekers)

        # Randomize layout
        self._randomize_layout()

        # Recompute BFS cache
        self._time_steps_to_goal_mapping = compute_time_steps_to_goal_mapping(
            goal_coordinates = self._runtime_goal,
            grid_dimensions = self.config.environment.grid_dimensions,
            obstacles_coordinates = self._runtime_obstacles,
            agent_velocity = self.config.agent.velocity
        )

        # Initialize seekers
        self._current_seekers_coordinates = [s.start_coordinates for s in self._runtime_seekers]
        self._current_step = 0

        # Initialize runners at random positions (or from config if not randomizing)
        occupied = set(self._runtime_obstacles) | {self._runtime_goal} | set(self._current_seekers_coordinates)
        for agent_id in self.possible_agents:
            if self._randomization and self._randomization.randomize_positions:
                from src.utils.environment.randomization import sample_free_cell
                start = sample_free_cell(self._rng, self.config.environment.grid_dimensions, occupied)
            else:
                start = self.config.agent.start_coordinates
            self._runner_coordinates[agent_id] = start
            occupied.add(start)

            local_obs = self._compute_local_observation(start)
            history: deque[np.ndarray] = deque(maxlen=self.config.agent.observation_stack_depth)
            for _ in range(self.config.agent.observation_stack_depth):
                history.append(local_obs)
            self._runner_observation_histories[agent_id] = history
            self._runner_active[agent_id] = True
            self._runner_nonprogress_steps[agent_id] = 0
            self._runner_starting_time_steps[agent_id] = self._time_steps_to_goal_mapping.get(start)
            self._runner_min_distance_to_hazards[agent_id] = compute_distance_to_hazards(
                current_agent_coordinates = start,
                obstacles_coordinates = self._runtime_obstacles,
                grid_dimensions = self.config.environment.grid_dimensions,
                seekers_coordinates = self._current_seekers_coordinates
            )

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]]
    ]:
        """
        Performs a step for all active runner agents simultaneously.
        
        :param actions: Dictionary mapping agent IDs to actions.
        :return: Tuple of (observations, rewards, terminations, truncations, infos).
        """
        self._current_step += 1
        truncated_all = self._current_step >= self.config.environment.episode_time_limit

        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}

        # Process each active runner
        for agent_id in self.agents:
            if not self._runner_active[agent_id]:
                rewards[agent_id] = 0.0
                terminations[agent_id] = True
                truncations[agent_id] = truncated_all
                continue

            action = actions.get(agent_id, 0)
            movement = self._action_map[action]
            coords = self._runner_coordinates[agent_id]

            # Previous state
            prev_time = self._time_steps_to_goal_mapping.get(coords)
            visible_before = compute_visible_seekers(
                current_agent_coordinates = coords,
                current_seekers_coordinates = self._current_seekers_coordinates,
                visibility_radius = self.config.agent.visibility_radius,
                visibility_rays = self._visibility_rays,
                obstacles_coordinates = self._runtime_obstacles
            )
            progress_coeff = 0.05 if visible_before else 0.1

            # Move agent
            updated, last_valid, collided, goal = move_agent(
                current_coordinates = coords,
                velocity = self.config.agent.velocity,
                movement_vector = movement,
                grid_dimensions = self.config.environment.grid_dimensions,
                obstacles_coordinates = self._runtime_obstacles,
                seekers_coordinates = frozenset(self._current_seekers_coordinates),
                goal_coordinates = self._runtime_goal
            )

            if collided:
                self._runner_coordinates[agent_id] = last_valid
                cur_time = self._time_steps_to_goal_mapping.get(last_valid)
            else:
                self._runner_coordinates[agent_id] = updated
                cur_time = self._time_steps_to_goal_mapping.get(updated)

            actual_coords = self._runner_coordinates[agent_id]

            # Visible seekers after move
            visible_after = compute_visible_seekers(
                current_agent_coordinates = actual_coords,
                current_seekers_coordinates = self._current_seekers_coordinates,
                visibility_radius = self.config.agent.visibility_radius,
                visibility_rays = self._visibility_rays,
                obstacles_coordinates = self._runtime_obstacles
            )

            # Nonprogress tracking
            if is_nonprogress_penalizable(
                previous_time_steps_to_goal = prev_time,
                current_time_steps_to_goal = cur_time,
                previously_visible_seekers_coordinates = visible_before
            ):
                self._runner_nonprogress_steps[agent_id] += 1
            else:
                self._runner_nonprogress_steps[agent_id] = 0

            # Distance to hazards
            cur_dist = compute_distance_to_hazards(
                current_agent_coordinates = actual_coords,
                obstacles_coordinates = self._runtime_obstacles,
                grid_dimensions = self.config.environment.grid_dimensions,
                seekers_coordinates = self._current_seekers_coordinates
            )
            self._runner_min_distance_to_hazards[agent_id] = compute_minimum_distance_to_hazards(
                current_minimum_distance_to_hazards = self._runner_min_distance_to_hazards[agent_id],
                current_distance_to_hazards = cur_dist
            )

            terminated = collided or goal
            reward = compute_rewards(
                goal = goal,
                collided = collided,
                intercepted = False,
                truncated = truncated_all,
                previous_time_steps_to_goal = prev_time,
                current_time_steps_to_goal = cur_time,
                progress_coefficient = progress_coeff,
                current_agent_coordinates = actual_coords,
                currently_visible_seekers_coordinates = visible_after,
                visibility_radius = self.config.agent.visibility_radius,
                num_consecutive_penalizable_nonprogress_steps = self._runner_nonprogress_steps[agent_id]
            )

            if terminated:
                self._runner_active[agent_id] = False

            rewards[agent_id] = reward
            terminations[agent_id] = terminated
            truncations[agent_id] = truncated_all

        # Move seekers (they chase the nearest active runner)
        active_runners = [a for a in self.agents if self._runner_active[a]]
        if active_runners:
            # Seekers chase the nearest active runner
            for i, seeker in enumerate(self._runtime_seekers):
                # Find nearest active runner to this seeker
                nearest_runner = min(
                    active_runners,
                    key=lambda a: max(
                        abs(self._current_seekers_coordinates[i].x - self._runner_coordinates[a].x),
                        abs(self._current_seekers_coordinates[i].y - self._runner_coordinates[a].y)
                    )
                )
                other_seekers = frozenset(
                    self._current_seekers_coordinates[:i] + self._current_seekers_coordinates[i+1:]
                )
                self._current_seekers_coordinates[i] = move_seeker(
                    current_coordinates = self._current_seekers_coordinates[i],
                    velocity = seeker.velocity,
                    policy = seeker.policy,
                    current_agent_coordinates = self._runner_coordinates[nearest_runner],
                    grid_dimensions = self.config.environment.grid_dimensions,
                    obstacles_coordinates = self._runtime_obstacles,
                    other_seekers_coordinates = other_seekers,
                    goal_coordinates = self._runtime_goal
                )

        # Check for interceptions after seeker movement
        for agent_id in self.agents:
            if not self._runner_active[agent_id]:
                continue
            if self._runner_coordinates[agent_id] in self._current_seekers_coordinates:
                self._runner_active[agent_id] = False
                rewards[agent_id] = compute_rewards(
                    goal = False,
                    collided = False,
                    intercepted = True,
                    truncated = truncated_all,
                    previous_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._runner_coordinates[agent_id]),
                    current_time_steps_to_goal = self._time_steps_to_goal_mapping.get(self._runner_coordinates[agent_id]),
                    progress_coefficient = 0.1,
                    current_agent_coordinates = self._runner_coordinates[agent_id],
                    currently_visible_seekers_coordinates = [],
                    visibility_radius = self.config.agent.visibility_radius,
                    num_consecutive_penalizable_nonprogress_steps = self._runner_nonprogress_steps[agent_id]
                )
                terminations[agent_id] = True

        # Handle truncation
        if truncated_all:
            for agent_id in self.agents:
                self._runner_active[agent_id] = False
                truncations[agent_id] = True

        # Update observations
        for agent_id in self.agents:
            local_obs = self._compute_local_observation(self._runner_coordinates[agent_id])
            self._runner_observation_histories[agent_id].append(local_obs)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Remove terminated/truncated agents
        self.agents = [a for a in self.agents if not (terminations.get(a, False) or truncations.get(a, False))]

        return observations, rewards, terminations, truncations, infos
