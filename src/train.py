from datetime import datetime, timezone
from typing import Callable, Literal, Sequence
from gymnasium import Env
from numpy import info
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor
from configs.experiments.reactive_avoidance import (
    LOGGING_DIRECTORY, EXPERIMENT_NAME, SEED, MODEL, SYSTEM_CONFIGURATIONS, TRAINING_METADATA, LOGGING_METADATA
)
from src.environment import ReactiveAvoidanceEnv, make_env_from_yaml, validate_environment
from src.utils.evaluation.helpers import evaluate_agent
from src.utils.logging.helpers import initialize_logging_directories, get_runtime_metadata, record_video_single_episode
from src.utils.logging.experiment_metadata import ExperimentMetadata
from src.utils.training.callbacks import RollingMetricsCallback, VideoCallback
from src.utils.typing.agent import StableBaselines3Agent
from src.utils.typing.environment import VideoRecordableEnvironmentFactory


def make_video_recordable_environment_factory(yaml_file_path: str) -> VideoRecordableEnvironmentFactory[ReactiveAvoidanceEnv]:
    """
    Creates a factory for creating video recordable environments.
    
    :param yaml_file_path: The path to the YAML file defining the environment.
    :return: A factory for creating video recordable environments from the YAML file.
    """
    def video_recordable_environment_factory(*, render_mode: Literal["rgb_array"]) -> ReactiveAvoidanceEnv:
        """
        Creates a video recordable environment.
        
        :param render_mode: The render mode to use for the environment. Must be "rgb_array".
        :return: A video recordable environment.
        """
        return make_env_from_yaml(yaml_file_path, render_mode = render_mode)
    return video_recordable_environment_factory


def make_vectorized_environment(*, yaml_file_path: str, n_envs: int, seed: int) -> VecEnv:
    """
    Creates a vectorized environment without auto-monitor wrapping around each environment.
    
    :param yaml_file_path: The path to the YAML file defining the environment.
    :param n_envs: The number of environments to create.
    :param seed: The seed to use for the environment.
    :return: A vectorized environment.
    """
    environment_factories = [lambda: make_env_from_yaml(yaml_file_path) for _ in range(n_envs)]
    vectorized_environment_class = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vectorized_environment = vectorized_environment_class(list(environment_factories))
    vectorized_environment.seed(seed)
    return vectorized_environment


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
    experiment_directory = initialize_logging_directories(logging_directory = LOGGING_DIRECTORY, experiment_metadata = experiment_metadata)

    # Validate training and evaluation environments
    validate_environment(SYSTEM_CONFIGURATIONS.training.path)
    validate_environment(SYSTEM_CONFIGURATIONS.validation.path)
    validate_environment(SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path)
    validate_environment(SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path)

    # Record a video of one episode of random actions in the training environment    
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/"),
        video_name_prefix = "random",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed
    )

    # Create vectorized training environment
    environment = make_vectorized_environment(
        yaml_file_path = SYSTEM_CONFIGURATIONS.training.path,
        n_envs = experiment_metadata.training.n_envs,
        seed = experiment_metadata.seed
    )
    environment = VecMonitor(
        environment,
        filename = str(experiment_directory / "training/metrics/monitor.csv"),
        info_keywords = ("outcome",)
    )

    # Create the PPO agent
    model = PPO(
        **experiment_metadata.model.hyperparameters,
        env = environment,
        stats_window_size = experiment_metadata.logging.rolling_window_size,
        tensorboard_log = str(experiment_directory / experiment_metadata.logging.logging_directories.tensorboard),
        verbose = experiment_metadata.logging.verbose,
        seed = experiment_metadata.seed,
        device = experiment_metadata.runtime.hardware.device
    )

    # Record a video of one episode of a PPO agent's actions in the training environment (before training)
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/pretraining"),
        video_name_prefix = "untrained",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(model)
    )

    # Define callbacks
    # Periodically save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq = max(1, (experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_checkpoints) // experiment_metadata.training.n_envs),
        save_path = str(experiment_directory / "models/checkpoints"),
        name_prefix = f"{experiment_metadata.experiment_name}_{experiment_metadata.model.name}",
        verbose = experiment_metadata.logging.verbose
    )
    # Periodically evaluate the agent on the validation environment; save the best model
    validation_environment = make_vectorized_environment(
        yaml_file_path = SYSTEM_CONFIGURATIONS.validation.path,
        n_envs = 1,
        seed = experiment_metadata.seed
    )
    validation_environment = VecMonitor(
        validation_environment,
        filename = None,
        info_keywords = ("outcome",)
    )
    eval_callback = EvalCallback(
        eval_env = validation_environment,
        n_eval_episodes = experiment_metadata.logging.episodes_per_validation_evaluation,
        eval_freq = max(1, (experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_validation_evaluations) // experiment_metadata.training.n_envs),
        log_path = str(experiment_directory / "validation/metrics"),
        best_model_save_path = str(experiment_directory / "models"),
        verbose = experiment_metadata.logging.verbose
    )
    # Periodically record a video of one episode of the PPO agent's actions in the training environment
    video_callback = VideoCallback(
        video_directory = str(experiment_directory / "training/videos/training"),
        video_name_prefix = "training",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        recording_frequency = experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_videos,
        verbose = experiment_metadata.logging.verbose
    )
    # Record rolling metrics via the Stable Baselines 3 logger
    rolling_metrics_callback = RollingMetricsCallback(
        rolling_window_size = experiment_metadata.logging.rolling_window_size,
        verbose = experiment_metadata.logging.verbose
    )
    callback_list = CallbackList([checkpoint_callback, eval_callback, video_callback, rolling_metrics_callback])

    # Train the PPO agent    
    model.learn(
        total_timesteps = experiment_metadata.training.total_timesteps,
        callback = callback_list,
        tb_log_name = f"{experiment_metadata.experiment_name}_{experiment_metadata.model.name}",
        progress_bar = True
    )

    # Save the final PPO agent
    model.save(str(experiment_directory / "models" / "final_model"))

    # Record a video of one episode of the learned PPO agent's actions in the training environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining"),
        video_name_prefix = "final",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(model)
    )

    # Load the best PPO agent
    best_model_path = experiment_directory / "models" / "best_model.zip"
    best_model = model
    if best_model_path.exists():
        best_model = PPO.load(str(best_model_path))
    if best_model == model:
        print("[Evaluation] No best model found. Using final model.")
    best_agent = StableBaselines3Agent(best_model)

    # Record a video of one episode of the best PPO agent's actions in the training environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )

    # Evaluate the best PPO agent on the in-distribution evaluation environment
    evaluate_agent(
        model = best_model,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path,
        environment_name = "In Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/in_distribution/metrics"
    )
    # Record a video of one episode of the best PPO agent's actions in the in-distribution evaluation environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/in_distribution/videos"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )
    
    # Evaluate the best PPO agent on the out-of-distribution evaluation environment
    evaluate_agent(
        model = best_model,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path,
        environment_name = "Out of Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/out_of_distribution/metrics"
    )
    
    # Record a video of one episode of the best PPO agent's actions in the out-of-distribution evaluation environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/out_of_distribution/videos"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )

    # Close the environment
    validation_environment.close()
    environment.close()


if __name__ == "__main__":
    main()
