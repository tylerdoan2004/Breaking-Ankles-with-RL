"""
This module provides an A* baseline agent that navigates to the goal using BFS-optimal pathfinding.
The agent treats walls and currently visible seekers as obstacles but makes no proactive effort to evade
seekers beyond what is forced. This serves as a baseline for comparing against the seeker-aware PPO agent.
"""
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from configs.experiments.reactive_avoidance import (
    LOGGING_DIRECTORY, SEED, SYSTEM_CONFIGURATIONS, LOGGING_METADATA
)
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.environment_creation import make_env_from_yaml, make_video_recordable_environment_factory
from src.utils.environment.helpers import compute_time_steps_to_goal_mapping, move_agent, MOVEMENT_VECTORS
from src.utils.logging.helpers import INFO_KEYWORDS, record_video_single_episode
from src.utils.typing.agent import StableBaselines3Agent
from src.utils.yaml_parser.configuration import SystemConfiguration


class AStarAgent:
    """
    An agent that navigates to the goal using BFS-optimal (A*) pathfinding.
    Treats walls and currently visible seeker positions as obstacles.
    Makes no proactive effort to evade seekers beyond the current timestep.
    """
    def __init__(self, config: SystemConfiguration) -> None:
        self._config = config
        self._time_steps_to_goal_mapping = compute_time_steps_to_goal_mapping(
            goal_coordinates = config.agent.goal_coordinates,
            grid_dimensions = config.environment.grid_dimensions,
            obstacles_coordinates = config.environment.obstacles_coordinates,
            agent_velocity = config.agent.velocity
        )
        self._width = config.environment.grid_dimensions.width
        self._height = config.environment.grid_dimensions.height
        self._visibility_radius = config.agent.visibility_radius
        self._visibility_diameter = 2 * self._visibility_radius + 1
        self._observation_stack_depth = config.agent.observation_stack_depth
        self._dims_per_local_obs = 4 * self._visibility_diameter ** 2

    def _decode_coordinates(self, scaled_x: float, scaled_y: float) -> Coordinates:
        x = round((scaled_x + 1.0) * (self._width - 1) / 2.0)
        y = round((scaled_y + 1.0) * (self._height - 1) / 2.0)
        return Coordinates(int(x), int(y))

    def _decode_visible_seekers(self, obs_1d: np.ndarray, current_coordinates: Coordinates) -> frozenset[Coordinates]:
        """
        Decodes visible seeker positions from the most recent local observation in the stack.
        The local observation stack starts at index 4 (after the 2D position and 2D goal vector).
        Observations are ordered oldest-to-newest, so the most recent frame is the last one.
        Channel 3 of the local observation has a 1.0 at cell (i, j) if a seeker is visible at
        offset (i - r, j - r) relative to the agent.
        """
        r = self._visibility_radius
        d = self._visibility_diameter
        # Jump to the start of the most recent local observation (last in the stack)
        start = 4 + (self._observation_stack_depth - 1) * self._dims_per_local_obs
        local_obs = obs_1d[start : start + self._dims_per_local_obs].reshape(4, d, d)
        seeker_channel = local_obs[3]
        visible_seekers: set[Coordinates] = set()
        for i in range(d):
            for j in range(d):
                if seeker_channel[i, j] == 1.0:
                    dx = i - r
                    dy = j - r
                    visible_seekers.add(Coordinates(current_coordinates.x + dx, current_coordinates.y + dy))
        return frozenset(visible_seekers)

    def _best_action(self, current_coordinates: Coordinates, visible_seekers: frozenset[Coordinates]) -> int:
        """
        Returns the action index that begins the BFS-optimal path to the goal.
        Treats walls and currently visible seeker positions as obstacles.
        """
        best_action = 0
        best_time_steps: int | float = float("inf")
        for action_idx, movement_vector in enumerate(MOVEMENT_VECTORS):
            next_coordinates, _, collided, goal = move_agent(
                current_coordinates = current_coordinates,
                velocity = self._config.agent.velocity,
                movement_vector = movement_vector,
                grid_dimensions = self._config.environment.grid_dimensions,
                obstacles_coordinates = self._config.environment.obstacles_coordinates,
                seekers_coordinates = visible_seekers,
                goal_coordinates = self._config.agent.goal_coordinates
            )
            if collided:
                continue
            if goal:
                return action_idx
            time_steps = self._time_steps_to_goal_mapping.get(next_coordinates, float("inf"))
            if time_steps < best_time_steps:
                best_time_steps = time_steps
                best_action = action_idx
        return best_action

    def predict(self, observation: np.ndarray, state: Any = None,
                episode_start: Any = None, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """
        Implements the StableBaselines3Model predict interface.
        Accepts observation of shape (obs_dim,) or (n_envs, obs_dim).
        Returns (actions, None) where actions has shape (n_envs,).
        """
        obs = np.asarray(observation, dtype = np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        n_envs = obs.shape[0]
        actions = np.empty(n_envs, dtype = np.int64)
        for i in range(n_envs):
            current_coordinates = self._decode_coordinates(float(obs[i, 0]), float(obs[i, 1]))
            visible_seekers = self._decode_visible_seekers(obs[i], current_coordinates)
            actions[i] = self._best_action(current_coordinates, visible_seekers)
        return actions, None


def evaluate_a_star(*, agent: AStarAgent, yaml_file_path: str, environment_name: str,
                    seed: int, n_episodes: int, metrics_directory: Path,
                    info_keywords: tuple[str, ...]) -> None:
    """
    Evaluates the A* agent on the environment for a given number of episodes.
    Mirrors evaluate_agent from src/utils/evaluation/helpers.py.
    """
    environment = DummyVecEnv([lambda: make_env_from_yaml(yaml_file_path)])
    environment.seed(seed)
    environment = VecMonitor(
        environment,
        filename = str(metrics_directory / "monitor.csv"),
        info_keywords = info_keywords
    )
    mean_reward, std_reward = evaluate_policy(
        agent,
        environment,
        n_eval_episodes = n_episodes,
        deterministic = True,
        render = False
    )
    environment.close()
    print(f"[Evaluation] {environment_name}")
    print(f"Number of episodes: {n_episodes}")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Standard deviation: {std_reward:.3f}")


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    seed = SEED
    n_episodes = LOGGING_METADATA.episodes_per_evaluation

    experiment_directory = Path(LOGGING_DIRECTORY) / "baseline" / "a_star" / timestamp
    for subdir in [
        "evaluation/in_distribution/metrics",
        "evaluation/in_distribution/videos",
        "evaluation/out_of_distribution/metrics",
        "evaluation/out_of_distribution/videos",
    ]:
        (experiment_directory / subdir).mkdir(parents = True, exist_ok = True)

    in_dist_path = SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path
    out_dist_path = SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path

    print("Evaluating A* baseline on in-distribution environment...")
    print()
    in_dist_agent = AStarAgent(SystemConfiguration.parse_config_file(Path(in_dist_path)))
    evaluate_a_star(
        agent = in_dist_agent,
        yaml_file_path = in_dist_path,
        environment_name = "In Distribution",
        seed = seed,
        n_episodes = n_episodes,
        metrics_directory = experiment_directory / "evaluation/in_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")

    print("Recording video of A* baseline in in-distribution environment...")
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/in_distribution/videos"),
        video_name_prefix = "a_star",
        environment_factory = make_video_recordable_environment_factory(in_dist_path),
        seed = seed,
        agent = StableBaselines3Agent(in_dist_agent)
    )
    print("Video recorded.")

    print("Evaluating A* baseline on out-of-distribution environment...")
    print()
    out_dist_agent = AStarAgent(SystemConfiguration.parse_config_file(Path(out_dist_path)))
    evaluate_a_star(
        agent = out_dist_agent,
        yaml_file_path = out_dist_path,
        environment_name = "Out of Distribution",
        seed = seed,
        n_episodes = n_episodes,
        metrics_directory = experiment_directory / "evaluation/out_of_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")

    print("Recording video of A* baseline in out-of-distribution environment...")
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/out_of_distribution/videos"),
        video_name_prefix = "a_star",
        environment_factory = make_video_recordable_environment_factory(out_dist_path),
        seed = seed,
        agent = StableBaselines3Agent(out_dist_agent)
    )
    print("Video recorded.")


if __name__ == "__main__":
    main()
