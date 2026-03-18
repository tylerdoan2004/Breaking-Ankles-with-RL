"""
A configuration file for multi-agent reactive avoidance experiments.
Uses 2 cooperative runners with randomized environments.
"""
from src.utils.logging.experiment_metadata import (
    EvaluationMetadata,
    LoggingDirectoriesMetadata,
    LoggingMetadata,
    ModelMetadata,
    SystemConfigurationMetadata,
    SystemConfigurationsMetadata,
    TrainingMetadata
)


LOGGING_DIRECTORY = "logs"
EXPERIMENT_NAME = "multi_agent_reactive_avoidance"
SEED = 0
NUM_RUNNERS = 2
MODEL = ModelMetadata(
    name = "ppo",
    hyperparameters = {
        "policy": "MlpPolicy",
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "policy_kwargs": {"net_arch": [128, 128]}
    }
)
SYSTEM_CONFIGURATIONS = SystemConfigurationsMetadata(
    training = SystemConfigurationMetadata(
        path = "configs/system/training/randomized_training.yaml"
    ),
    validation = SystemConfigurationMetadata(
        path = "configs/system/validation/validation.yaml"
    ),
    evaluation = EvaluationMetadata(
        in_distribution = SystemConfigurationMetadata(
            path = "configs/system/evaluation/in_distribution.yaml"
        ),
        out_of_distribution = SystemConfigurationMetadata(
            path = "configs/system/evaluation/out_of_distribution.yaml"
        )
    )
)
TRAINING_METADATA = TrainingMetadata(
    n_envs = 4,
    # Multi-agent needs more steps: 5M total
    total_timesteps = 5_013_504
)
LOGGING_METADATA = LoggingMetadata(
    logging_directories = LoggingDirectoriesMetadata(
        base = LOGGING_DIRECTORY,
        tensorboard = "training/metrics/tensorboard"
    ),
    verbose = 1,
    rolling_window_size = 100,
    num_checkpoints = 10,
    num_validation_evaluations = 10,
    episodes_per_validation_evaluation = 5,
    episodes_per_evaluation = 10,
    num_videos = 5
)
