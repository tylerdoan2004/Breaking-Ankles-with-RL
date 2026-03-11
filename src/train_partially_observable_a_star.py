"""
This module provides a deterministic, partially-observable agent that navigates to the goal coordinates using A* pathfinding. This module provides the main rollout and evaluation loop for the partially-observable A* agent. The agent serves as a baseline for comparing against the PPO agent.
"""
import numpy as np
import yaml
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from gymnasium import Env
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter
from configs.experiments.partially_observable_a_star import (
    EXPERIMENT_NAME, LOGGING_DIRECTORY, MODEL, SEED, SYSTEM_CONFIGURATIONS, LOGGING_METADATA, TRAINING_METADATA
)
from src.utils.environment.coordinates import Coordinates
from src.utils.environment.environment_creation import make_vectorized_environment, make_video_recordable_environment_factory, validate_environment
from src.utils.environment.helpers import compute_time_steps_to_goal_mapping, move_agent, MOVEMENT_VECTORS
from src.utils.environment.vector import Vector
from src.utils.evaluation.helpers import evaluate_agent
from src.utils.logging.experiment_metadata import ExperimentMetadata
from src.utils.logging.helpers import INFO_KEYWORDS, get_runtime_metadata, initialize_logging_directories, record_video_single_episode
from src.utils.typing.agent import StableBaselines3Agent
from src.utils.yaml_parser.configuration import SystemConfiguration


class PartiallyObservableAStarAgent:
    """
    A deterministic, partially-observable agent that navigates to the goal coordinates using A* pathfinding.
    
    The agent decodes the same observation vector consumed by the PPO policy, uses only currently visible seekers, and chooses the action that begins the shortest path to the goal while treating currently visible seekers as blocked cells.
    """
    def __init__(self, config: SystemConfiguration) -> None:
        """
        Initializes the deterministic, partially-observable A* agent.

        :param config: A SystemConfiguration object representing the configuration of the agent-seeker-gridworld system.
        :return: None.
        """
        self.config = config
        # Compute caches
        self._time_steps_to_goal_mapping = compute_time_steps_to_goal_mapping(
            goal_coordinates = config.agent.goal_coordinates,
            grid_dimensions = config.environment.grid_dimensions,
            obstacles_coordinates = config.environment.obstacles_coordinates,
            agent_velocity = config.agent.velocity
        )
        # NOTE: Refer to calculate_observation_space in src/utils/environment/helpers.py and _compute_local_observation in src/environment.py for details on the observation vector format
        self.visibility_radius = self.config.agent.visibility_radius
        self.visibility_diameter = 2 * self.visibility_radius + 1
        current_observation_dimensions = 4
        self.observation_channels = 4
        dimensions_per_local_observation = self.observation_channels * self.visibility_diameter ** 2
        self.current_local_observation_starting_index = current_observation_dimensions + (self.config.agent.observation_stack_depth - 1) * dimensions_per_local_observation

    def _decode_coordinates(self, x_scaled: float, y_scaled: float) -> Coordinates:
        """
        Decodes a pair of scaled coordinates into a Coordinates object.

        :param x_scaled: The scaled x-coordinate. Should be in the range [-1, 1].
        :param y_scaled: The scaled y-coordinate. Should be in the range [-1, 1].
        :return: A Coordinates object representing the decoded coordinates.
        """
        x = round((x_scaled + 1.0) * (self.config.environment.grid_dimensions.width - 1) / 2.0)
        y = round((y_scaled + 1.0) * (self.config.environment.grid_dimensions.height - 1) / 2.0)
        return Coordinates(int(x), int(y))

    def _decode_visible_seekers(self, observation: np.ndarray, current_coordinates: Coordinates) -> frozenset[Coordinates]:
        """
        Decodes currently visible seekers from the most recent local observation.

        :param observation: The one-dimensional observation vector.
        :param current_coordinates: The current coordinates of the agent.
        :return: A frozenset of Coordinates representing the locations of currently visible seekers.
        """
        current_local_observation = observation[self.current_local_observation_starting_index:]
        current_local_observation = current_local_observation.reshape(self.observation_channels,
                                                                      self.visibility_diameter,
                                                                      self.visibility_diameter)
        # NOTE: Refer to calculate_observation_space in src/utils/environment/helpers.py and _compute_local_observation in src/environment.py for details on the observation vector format
        seekers_channel = current_local_observation[3]
        
        visible_seekers: set[Coordinates] = set()
        for i in range(self.visibility_diameter):
            for j in range(self.visibility_diameter):
                if seekers_channel[i, j] != 1.0:
                    continue
                dx = i - self.visibility_radius
                dy = j - self.visibility_radius
                visible_seekers.add(current_coordinates + Vector(dx, dy))
        return frozenset(visible_seekers)

    # TODO: Validate _best_action
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
                velocity = self.config.agent.velocity,
                movement_vector = movement_vector,
                grid_dimensions = self.config.environment.grid_dimensions,
                obstacles_coordinates = self.config.environment.obstacles_coordinates,
                seekers_coordinates = visible_seekers,
                goal_coordinates = self.config.agent.goal_coordinates
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

    # TODO: Validate predict
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

    def save(self, file_path: Path, experiment_metadata: ExperimentMetadata) -> None:
        """
        Saves the agent to a YAML file.

        :param file_path: The path to save the agent to. The file name should contain no extension.
        :param experiment_metadata: The experiment metadata.
        :return: None.
        """
        file_path.parent.mkdir(parents = True, exist_ok = True)
        payload = {
            "name": experiment_metadata.model.name,
            "hyperparameters": experiment_metadata.model.hyperparameters,
            "visibility_radius": self.config.agent.visibility_radius
        }
        with open(file_path.with_suffix(".yaml"), "w", encoding = "utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys = False, allow_unicode = True)


# TODO: Validate _maybe_record_training_videos
def _maybe_record_training_videos(
    *,
    num_timesteps: int,
    next_recording_timestep: Optional[int],
    recording_frequency: int,
    video_directory: Path,
    environment_factory,
    agent: PartiallyObservableAStarAgent,
    seed: int,
) -> Optional[int]:
    if next_recording_timestep is None:
        return None
    while num_timesteps >= next_recording_timestep:
        record_video_single_episode(
            video_directory = str(video_directory),
            video_name_prefix = f"training-{next_recording_timestep}-steps",
            environment_factory = environment_factory,
            seed = seed,
            agent = StableBaselines3Agent(agent),
        )
        next_recording_timestep += recording_frequency
    return next_recording_timestep


# TODO: Validate _rollout_training_agent
def _rollout_training_agent(
    *,
    agent: PartiallyObservableAStarAgent,
    environment: Env,
    total_timesteps: int,
    tensorboard_directory: Path,
    rolling_window_size: int,
    recording_frequency: int,
    training_video_directory: Path,
    environment_factory,
    seed: int,
) -> None:
    """
    Runs the fixed A* agent in the vectorized training environment long enough to produce
    monitor.csv data, TensorBoard scalars, and periodic training videos.
    """
    writer = SummaryWriter(log_dir = str(tensorboard_directory))

    rolling_rewards: deque[float] = deque(maxlen = rolling_window_size)
    rolling_episode_lengths: deque[float] = deque(maxlen = rolling_window_size)
    rolling_outcomes: deque[str] = deque(maxlen = rolling_window_size)
    rolling_numeric_metrics: dict[str, deque[float]] = {
        key: deque(maxlen = rolling_window_size)
        for key in (
            "net_progress",
            "path_efficiency",
            "minimum_distance_to_obstacle",
            "minimum_distance_to_boundary",
            "minimum_distance_to_seeker",
        )
    }
    rolling_collision_types: deque[str] = deque(maxlen = rolling_window_size)
    rolling_interceptor_policies: deque[str] = deque(maxlen = rolling_window_size)

    observation = environment.reset()
    num_timesteps = 0
    next_recording_timestep: Optional[int] = recording_frequency if recording_frequency > 0 else None

    while num_timesteps < total_timesteps:
        actions, _ = agent.predict(observation, deterministic = True)
        observation, rewards, dones, infos = environment.step(actions)
        del rewards  # VecMonitor writes rewards to monitor.csv and info["episode"] for finished episodes.
        num_timesteps += environment.num_envs

        finished_episode_infos = [info for info, done in zip(infos, dones) if done and isinstance(info, dict)]
        if finished_episode_infos:
            finished_rewards: list[float] = []
            finished_lengths: list[float] = []
            numeric_values_this_step: dict[str, list[float]] = {key: [] for key in rolling_numeric_metrics}

            for info in finished_episode_infos:
                episode_info = info.get("episode")
                if isinstance(episode_info, dict):
                    reward = episode_info.get("r")
                    length = episode_info.get("l")
                    if isinstance(reward, (int, float)):
                        finished_rewards.append(float(reward))
                        rolling_rewards.append(float(reward))
                    if isinstance(length, (int, float)):
                        finished_lengths.append(float(length))
                        rolling_episode_lengths.append(float(length))

                outcome = info.get("outcome")
                if isinstance(outcome, str):
                    rolling_outcomes.append(outcome)

                for key in rolling_numeric_metrics:
                    value = info.get(key)
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)
                        numeric_values_this_step[key].append(numeric_value)
                        rolling_numeric_metrics[key].append(numeric_value)

                collision_type = info.get("collision_type")
                if isinstance(collision_type, str):
                    rolling_collision_types.append(collision_type)

                interceptor_policy = info.get("interceptor_policy")
                if isinstance(interceptor_policy, str):
                    rolling_interceptor_policies.append(interceptor_policy)

            if rolling_rewards:
                writer.add_scalar("rollout/ep_rew_mean", sum(rolling_rewards) / len(rolling_rewards), num_timesteps)
            if rolling_episode_lengths:
                writer.add_scalar("rollout/ep_len_mean", sum(rolling_episode_lengths) / len(rolling_episode_lengths), num_timesteps)

            for key, values in numeric_values_this_step.items():
                if values:
                    writer.add_scalar(f"episode/{key}", sum(values) / len(values), num_timesteps)

            if rolling_outcomes:
                counts = Counter(rolling_outcomes)
                total = len(rolling_outcomes)
                for outcome in ("goal", "collision", "interception", "timeout"):
                    writer.add_scalar(f"outcomes/{outcome}_rate", counts.get(outcome, 0) / total, num_timesteps)

            for key, values in rolling_numeric_metrics.items():
                if values:
                    writer.add_scalar(f"rolling/{key}_mean", sum(values) / len(values), num_timesteps)

            if rolling_collision_types:
                counts = Counter(rolling_collision_types)
                total = len(rolling_collision_types)
                for collision_type in ("obstacle", "boundary", "seeker"):
                    writer.add_scalar(f"collision_type/{collision_type}_rate", counts.get(collision_type, 0) / total, num_timesteps)

            if rolling_interceptor_policies:
                counts = Counter(rolling_interceptor_policies)
                total = len(rolling_interceptor_policies)
                for interceptor_policy in ("random", "greedy", "a-star"):
                    writer.add_scalar(
                        f"interceptor_policy/{interceptor_policy}_rate",
                        counts.get(interceptor_policy, 0) / total,
                        num_timesteps,
                    )

        next_recording_timestep = _maybe_record_training_videos(
            num_timesteps=num_timesteps,
            next_recording_timestep=next_recording_timestep,
            recording_frequency=recording_frequency,
            video_directory=training_video_directory,
            environment_factory=environment_factory,
            agent=agent,
            seed=seed,
        )

    writer.flush()
    writer.close()


# TODO: Add "validation" metrics (although partially-observable A* agent is deterministic)
def main():
    experiment_metadata = ExperimentMetadata(
        experiment_name = EXPERIMENT_NAME,
        timestamp = datetime.now(timezone.utc),
        seed = SEED,
        model = MODEL,
        system_configurations = SYSTEM_CONFIGURATIONS,
        training = TRAINING_METADATA,
        runtime = get_runtime_metadata(),
        logging = LOGGING_METADATA
    )
    print("Initializing logging directories...")
    experiment_directory = initialize_logging_directories(logging_directory = LOGGING_DIRECTORY,
                                                         experiment_metadata = experiment_metadata)
    print("Logging directories initialized.")

    # Validate training and evaluation environments
    print("Validating environments...")
    validate_environment(SYSTEM_CONFIGURATIONS.training.path)
    validate_environment(SYSTEM_CONFIGURATIONS.validation.path)
    validate_environment(SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path)
    validate_environment(SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path)
    print("Environments validated.")

    print("Recording video of random actions in the training environment...")
    # Record a video of one episode of random actions in the training environment    
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/"),
        video_name_prefix = "random",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed
    )
    print("Video recorded.")

    print("Creating vectorized training environment...")
    # Craete vectorized training environment
    environment = make_vectorized_environment(
        yaml_file_path = SYSTEM_CONFIGURATIONS.training.path,
        n_envs = experiment_metadata.training.n_envs,
        seed = experiment_metadata.seed
    )
    environment = VecMonitor(
        environment,
        filename = str(experiment_directory / "training/metrics/monitor.csv"),
        info_keywords = INFO_KEYWORDS,
    )
    print("Vectorized training environment created.")

    print("Creating partially observable A* agent...")
    # Create partially observable A* agent
    agent = PartiallyObservableAStarAgent(config = SystemConfiguration.parse_config_file(Path(SYSTEM_CONFIGURATIONS.training.path)))
    print("Partially observable A* agent created.")

    # NOTE: We record this video for logging consistency with experiments on the PPO agent
    print("Recording video of partially observable A* agent's actions in the training environment (before full rollout)...")
    # Record a video of one episode of a partially observable A* agent's actions in the training environment (before full rollout)
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/pretraining/"),
        video_name_prefix = "pretraining",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(agent)
    )
    print("Video recorded.")

    print("Performing full rollout of partially observable A* agent...")
    _rollout_training_agent(
        agent = agent,
        environment = environment,
        total_timesteps = experiment_metadata.training.total_timesteps,
        tensorboard_directory = experiment_directory / experiment_metadata.logging.logging_directories.tensorboard,
        rolling_window_size = experiment_metadata.logging.rolling_window_size,
        recording_frequency = max(1, 
                                 experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_videos),
        training_video_directory = experiment_directory / "training/videos/training",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed
    )
    print("Full rollout performed.")

    # NOTE: The partially observable A* agent has no learnable parameters, so the final model is equivalent to the initial model
    print("Saving final partially observable A* agent...")
    # Save the partially observable A* agent
    agent.save(experiment_directory / "models" / "final_model",
               experiment_metadata = experiment_metadata)
    print("Final partially observable A* agent saved.")

    # NOTE: We record this video for logging consistency with experiments on the PPO agent
    print("Recording video of partially observable A* agent's actions in the training environment (after full rollout)...")
    # Record a video of one episode of a partially observable A* agent's actions in the training environment (after full rollout)
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining/"),
        video_name_prefix = "final",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(agent)
    )
    print("Video recorded.")

    # NOTE: The partially observable A* agent has no learnable parameters, so the best model is equivalent to the initial model
    print("Saving best partially observable A* agent...")
    agent.save(experiment_directory / "models" / "best_model",
               experiment_metadata = experiment_metadata)
    print("Best partially observable A* agent saved.")

    # NOTE: The partially observable A* agent has no learnable parameters, so the best model is equivalent to the initial model
    print("Loading best partially observable A* agent...")
    best_agent = agent
    print("Best partially observable A* agent loaded.")

    # NOTE: We record this video for logging consistency with experiments on the PPO agent
    print("Recording video of best partially observable A* agent's actions in the training environment...")
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining/"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(best_agent)
    )
    print("Video recorded.")

    print("Evaluating best partially observable A* agent on in-distribution environment...")
    print()
    evaluate_agent(
        model = best_agent,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path,
        environment_name = "In Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/in_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")
    print("Recording video of best partially observable A* agent's actions in the in-distribution evaluation environment...")
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/in_distribution/videos/"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(best_agent)
    )
    print("Video recorded.")
    print("Evaluating best partially observable A* agent on out-of-distribution environment...")
    print()
    evaluate_agent(
        model = best_agent,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path,
        environment_name = "Out of Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/out_of_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")
    print("Recording video of best partially observable A* agent's actions in the out-of-distribution evaluation environment...")
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/out_of_distribution/videos/"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(best_agent)
    )
    print("Video recorded.")

    print("Performing cleanup...")
    environment.close()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
