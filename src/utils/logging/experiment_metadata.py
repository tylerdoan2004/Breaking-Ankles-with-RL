"""
This module provides the ExperimentMetadata class for representing metadata about an experiment.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen = True, kw_only = True)
class ExperimentMetadata:
    """
    A class for representing metadata about an experiment.
    """
    experiment_name: str
    model_name: str
    timestamp: datetime
    seed: int
    model_hyperparameters: dict
    config_paths: dict[str, Path]
