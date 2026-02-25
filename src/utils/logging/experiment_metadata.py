"""
This module provides the ExperimentMetadata class for representing metadata about an experiment.
"""
import yaml
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict


ConfigPaths = TypedDict(
    "ConfigPaths",
    {
        "training": str,
        "evaluation/in_distribution": str,
        "evaluation/out_of_distribution": str
    }
)


@dataclass(frozen = True, kw_only = True)
class ExperimentMetadata:
    """
    A class for representing metadata about an experiment.
    """
    experiment_name: str
    model_name: str
    timestamp: datetime
    seed: int
    model_hyperparameters: dict[str, Any]
    config_paths: ConfigPaths

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ExperimentMetadata object to a YAML-serializable dictionary.
        
        :return: A YAML-serializable dictionary representing the ExperimentMetadata object.
        """
        data = asdict(self)
        data["timestamp"] = data["timestamp"].isoformat()
        return data

    def to_yaml(self, file_path: str | Path) -> None:
        """
        Writes the ExperimentMetadata object to a YAML file.

        :param file_path: The path to the YAML file to write.
        :return: None.
        """
        with open(file_path, "w", encoding = "utf-8") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys = False, allow_unicode = True)
