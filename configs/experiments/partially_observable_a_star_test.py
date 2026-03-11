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
    name = "a-star",
    hyperparameters = {
        "policy": "partially_observable"
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
