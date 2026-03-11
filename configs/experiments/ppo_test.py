"""
A configuration file for testing a reactive avoidance experiment.
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
EXPERIMENT_NAME = "reactive_avoidance"
SEED = 0
MODEL = ModelMetadata(
    name = "ppo",
    # NOTE: Default model hyperparameters obtained from https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
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
        # NOTE: Adjusted from 0.0 to 0.005 to prevent premature convergence
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
        "sde_sample_freq": -1,
        "rollout_buffer_class": None,
        "rollout_buffer_kwargs": None,
        "target_kl": None,
        "policy_kwargs": None
    }
)
SYSTEM_CONFIGURATIONS = SystemConfigurationsMetadata(
    training = SystemConfigurationMetadata(
        path = "configs/system/training/training.yaml"
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
    n_envs = 2,
    # NOTE: To guarantee expected logging behavior, set total timesteps to a multiple of MODEL.hyperparameters.n_steps * n_envs
    total_timesteps = 4_096
)
LOGGING_METADATA = LoggingMetadata(
    logging_directories = LoggingDirectoriesMetadata(
        base = LOGGING_DIRECTORY,
        tensorboard = "training/metrics/tensorboard"
    ),
    verbose = 0,
    rolling_window_size = 4,
    num_checkpoints = 4,
    num_validation_evaluations = 4,
    episodes_per_validation_evaluation = 2,
    episodes_per_evaluation = 2,
    num_videos = 4
)
