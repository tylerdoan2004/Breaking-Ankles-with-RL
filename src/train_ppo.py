"""
This module provides the main training and evaluation loop for the PPO agent.
"""
from datetime import datetime, timezone
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from configs.experiments.ppo import (
    LOGGING_DIRECTORY, EXPERIMENT_NAME, SEED, MODEL, SYSTEM_CONFIGURATIONS, TRAINING_METADATA, LOGGING_METADATA
)
from src.utils.environment.environment_creation import make_vectorized_environment, make_video_recordable_environment_factory, validate_environment
from src.utils.evaluation.helpers import evaluate_agent
from src.utils.logging.helpers import (
    INFO_KEYWORDS, initialize_logging_directories, get_runtime_metadata, record_video_single_episode
)
from src.utils.logging.experiment_metadata import ExperimentMetadata
from src.utils.training.callbacks import RollingMetricsCallback, VideoCallback
from src.utils.typing.agent import StableBaselines3Agent


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
    experiment_directory = initialize_logging_directories(logging_directory = LOGGING_DIRECTORY, experiment_metadata = experiment_metadata)
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
    # Create vectorized training environment
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
    
    print("Creating PPO agent...")
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
    print("PPO agent created.")

    print("Recording video of PPO agent's actions in the training environment (before training)...")
    # Record a video of one episode of a PPO agent's actions in the training environment (before training)
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/pretraining"),
        video_name_prefix = "pretraining",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(model)
    )
    print("Video recorded.")

    # Define callbacks
    # Periodically save model checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq = max(1, (experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_checkpoints) // experiment_metadata.training.n_envs),
        save_path = str(experiment_directory / "models/checkpoints"),
        name_prefix = "training",
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
        filename = str(experiment_directory / "validation/metrics/monitor.csv"),
        info_keywords = INFO_KEYWORDS
    )
    eval_callback = EvalCallback(
        eval_env = validation_environment,
        n_eval_episodes = experiment_metadata.logging.episodes_per_validation_evaluation,
        eval_freq = max(1, (experiment_metadata.training.total_timesteps // experiment_metadata.logging.num_validation_evaluations) // experiment_metadata.training.n_envs),
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

    print("Training PPO agent...")
    # Train the PPO agent    
    model.learn(
        total_timesteps = experiment_metadata.training.total_timesteps,
        callback = callback_list,
        tb_log_name = f"{experiment_metadata.experiment_name}_{experiment_metadata.model.name}",
        progress_bar = True
    )
    print("PPO agent trained.")

    print("Saving final PPO agent...")
    # Save the final PPO agent
    model.save(str(experiment_directory / "models" / "final_model"))
    print("Final PPO agent saved.")

    print("Recording video of PPO agent's actions in the training environment (after training)...")
    # Record a video of one episode of the learned PPO agent's actions in the training environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining"),
        video_name_prefix = "final",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = StableBaselines3Agent(model)
    )
    print("Video recorded.")

    print("Loading best PPO agent...")
    # Load the best PPO agent
    best_model_path = experiment_directory / "models" / "best_model.zip"
    best_model = model
    if best_model_path.exists():
        best_model = PPO.load(str(best_model_path))
    if best_model == model:
        print("[Evaluation] No best model found. Using final model.")
    best_agent = StableBaselines3Agent(best_model)
    print("Best PPO agent loaded.")

    print("Recording video of best PPO agent's actions in the training environment...")
    # Record a video of one episode of the best PPO agent's actions in the training environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "training/videos/posttraining"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.training.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )
    print("Video recorded.")

    print("Evaluating best PPO agent on in-distribution environment...")
    print()
    # Evaluate the best PPO agent on the in-distribution evaluation environment
    evaluate_agent(
        model = best_model,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path,
        environment_name = "In Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/in_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")
    print("Recording video of best PPO agent's actions in the in-distribution evaluation environment...")
    # Record a video of one episode of the best PPO agent's actions in the in-distribution evaluation environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/in_distribution/videos"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.in_distribution.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )
    print("Video recorded.")
    print("Evaluating best PPO agent on out-of-distribution environment...")
    print()
    # Evaluate the best PPO agent on the out-of-distribution evaluation environment
    evaluate_agent(
        model = best_model,
        yaml_file_path = SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path,
        environment_name = "Out of Distribution",
        seed = experiment_metadata.seed,
        n_episodes = experiment_metadata.logging.episodes_per_evaluation,
        metrics_directory = experiment_directory / "evaluation/out_of_distribution/metrics",
        info_keywords = INFO_KEYWORDS
    )
    print()
    print("Evaluation complete.")
    print("Recording video of best PPO agent's actions in the out-of-distribution evaluation environment...")
    # Record a video of one episode of the best PPO agent's actions in the out-of-distribution evaluation environment
    record_video_single_episode(
        video_directory = str(experiment_directory / "evaluation/out_of_distribution/videos"),
        video_name_prefix = "best",
        environment_factory = make_video_recordable_environment_factory(SYSTEM_CONFIGURATIONS.evaluation.out_of_distribution.path),
        seed = experiment_metadata.seed,
        agent = best_agent
    )
    print("Video recorded.")
    
    print("Performing cleanup...")
    # Close the environment
    validation_environment.close()
    environment.close()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
