"""
This module provides the ExperimentMetadata class (and related classes) for representing metadata about an experiment.
"""
import yaml
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen = True, kw_only = True, slots = True)
class ModelMetadata:
    """
    A class for representing metadata about a model.
    """
    name: str
    hyperparameters: dict[str, Any]


@dataclass(frozen = True, kw_only = True, slots = True)
class SystemConfigurationMetadata:
    """
    A class for representing metadata about a system configuration.
    """
    path: str


@dataclass(frozen = True, kw_only = True, slots = True)
class EvaluationMetadata:
    """
    A class for representing metadata about the evaluation configurations of the experiment.
    """
    in_distribution: SystemConfigurationMetadata
    out_of_distribution: SystemConfigurationMetadata


@dataclass(frozen = True, kw_only = True, slots = True)
class SystemConfigurationsMetadata:
    """
    A class for representing metadata about the system configurations of the experiment.
    """
    training: SystemConfigurationMetadata
    evaluation: EvaluationMetadata


@dataclass(frozen = True, kw_only = True, slots = True)
class TrainingMetadata:
    """
    A class for representing metadata about the training of the experiment.
    """
    n_envs: int
    total_timesteps: int


@dataclass(frozen = True, kw_only = True, slots = True)
class HardwareMetadata:
    """
    A class for representing metadata about the hardware environment of the experiment.
    """
    platform: str
    device: str


@dataclass(frozen = True, kw_only = True, slots = True)
class PythonMetadata:
    """
    A class for representing metadata about the Python environment of the experiment.
    """
    implementation: str
    version: str


@dataclass(frozen = True, kw_only = True, slots = True)
class PackageMetadata:
    """
    A class for representing metadata about a package.
    """
    version: Optional[str]


@dataclass(frozen = True, kw_only = True, slots = True)
class SoftwareMetadata:
    """
    A class for representing metadata about the software environment of the experiment.
    """
    python: PythonMetadata
    gymnasium: PackageMetadata
    minigrid: PackageMetadata
    numpy: PackageMetadata
    pyyaml: PackageMetadata
    stable_baselines3: PackageMetadata


@dataclass(frozen = True, kw_only = True, slots = True)
class RuntimeMetadata:
    """
    A class for representing metadata about the runtime environment of the experiment.
    """
    hardware: HardwareMetadata
    software: SoftwareMetadata


@dataclass(frozen = True, kw_only = True, slots = True)
class LoggingDirectoriesMetadata:
    """
    A class for representing metadata about the logging directories of the experiment.
    """
    base: str
    tensorboard: str
    

@dataclass(frozen = True, kw_only = True, slots = True)
class LoggingMetadata:
    """
    A class for representing metadata about the logging environment of the experiment.
    """
    logging_directories: LoggingDirectoriesMetadata
    verbose: Optional[int] = None
    rolling_window_size: Optional[int] = None


def _make_yaml_safe(object: Any) -> Any:
    if object is None or isinstance(object, (str, int, float, bool)):
        return object
    if isinstance(object, dict):
        return {str(key): _make_yaml_safe(value) for key, value in object.items()}
    if isinstance(object, (list, tuple)):
        return [_make_yaml_safe(item) for item in object]
    if isinstance(object, datetime):
        return object.isoformat()
    if isinstance(object, Path):
        return str(object)
    return repr(object)


@dataclass(frozen = True, kw_only = True, slots = True)
class ExperimentMetadata:
    """
    A class for representing top-level metadata about an experiment.
    """
    experiment_name: str
    timestamp: datetime
    seed: int
    model: ModelMetadata
    system_configurations: SystemConfigurationsMetadata
    training: TrainingMetadata
    runtime: RuntimeMetadata
    logging: LoggingMetadata

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ExperimentMetadata object to a YAML-serializable dictionary.
        
        :return: A YAML-serializable dictionary representing the ExperimentMetadata object.
        """
        return _make_yaml_safe(asdict(self))

    def to_yaml(self, file_path: str | Path) -> None:
        """
        Writes the ExperimentMetadata object to a YAML file.

        :param file_path: The path to the YAML file to write.
        :return: None.
        """
        with open(file_path, "w", encoding = "utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys = False, allow_unicode = True)
