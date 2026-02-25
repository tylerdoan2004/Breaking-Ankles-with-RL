from datetime import datetime, timezone
from configs.experiments.reactive_avoidance import LOGGING_DIRECTORY, EXPERIMENT_NAME, SEED, MODEL, SYSTEM_CONFIGURATIONS, LOGGING_METADATA
from src.utils.logging.helpers import initialize_logging_directories, get_runtime_metadata
from src.utils.logging.experiment_metadata import ExperimentMetadata


def main():
    experiment_metadata = ExperimentMetadata(
        experiment_name = EXPERIMENT_NAME,
        timestamp = datetime.now(timezone.utc),
        seed = SEED,
        model = MODEL,
        system_configurations = SYSTEM_CONFIGURATIONS,
        runtime = get_runtime_metadata(),
        logging = LOGGING_METADATA
    )
    initialize_logging_directories(logging_directory = LOGGING_DIRECTORY, experiment_metadata = experiment_metadata)


if __name__ == "__main__":
    main()
