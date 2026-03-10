"""
This module provides callbacks for logging an agent's training progress.
"""
from collections import Counter, deque
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from src.utils.logging.helpers import record_video_single_episode
from src.utils.typing.agent import StableBaselines3Agent
from src.utils.typing.environment import VideoRecordableEnvironmentFactory


class VideoCallback(BaseCallback):
    """
    A callback for recording a video of a single episode of an agent performing actions in an environment.
    """
    def __init__(self, *, video_directory: str, video_name_prefix: str,
                 environment_factory: VideoRecordableEnvironmentFactory, seed: int,
                 recording_frequency: int,
                 verbose: int = 0) -> None:
        """
        Initializes the VideoCallback object.
        
        :param video_directory: The directory used to store the video.
        :param video_name_prefix: The prefix used to name the video.
        :param environment_factory: A function that creates a video recordable environment.
        :param seed: The seed to use for the environment.
        :param recording_frequency: The number of time steps between recordings.
        :param verbose: The verbosity level.
        :return: None.
        """
        super().__init__(verbose = verbose)
        self.video_directory = Path(video_directory)
        self.video_name_prefix = video_name_prefix
        self.environment_factory = environment_factory
        self.seed = seed
        self.recording_frequency = recording_frequency
        self.next_recording_timestep: Optional[int] = None

    def _init_callback(self) -> None:
        """
        Initializes the callback.

        :return: None.
        """
        self.video_directory.mkdir(parents = True, exist_ok = True)
        self.next_recording_timestep = self.recording_frequency

    def _on_step(self) -> bool:
        """
        Implements the _on_step method of the BaseCallback class.
        
        :return: Whether to continue training. Always returns True.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        Records a video of a single episode of an agent performing actions in an environment.
        
        :return: None.
        """
        assert self.model is not None
        assert self.next_recording_timestep is not None
    
        while self.num_timesteps >= self.next_recording_timestep:
            try:
                record_video_single_episode(
                    video_directory = str(self.video_directory),
                    video_name_prefix = f"{self.video_name_prefix}-{self.next_recording_timestep}-steps",
                    environment_factory = self.environment_factory,
                    agent = StableBaselines3Agent(self.model),
                    seed = self.seed
                )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[VideoCallback] Failed to record video at timestep {self.next_recording_timestep}: {e}")
            finally:
                self.next_recording_timestep += self.recording_frequency


class RollingMetricsCallback(BaseCallback):
    """
    A callback for computing and logging rolling metrics via the Stable Baselines3 logger.
    """
    def __init__(self, *,
                 rolling_window_size: int,
                 outcomes: tuple[str, ...] = ("goal",
                                              "collision",
                                              "interception",
                                              "timeout"),
                 numeric_info_keywords: tuple[str, ...] = ("net_progress",
                                                           "path_efficiency",
                                                           "minimum_distance_to_obstacle",
                                                           "minimum_distance_to_boundary",
                                                           "minimum_distance_to_seeker"),
                 collision_types: tuple[str, ...] = ("obstacle",
                                                     "boundary",
                                                     "seeker"),
                 verbose: int = 0) -> None:
        """
        Initializes the RollingMetricsCallback object.
        
        :param rolling_window_size: The size of the rolling window used to compute rolling metrics.
        :param outcomes: Possible episode outcomes stored in the info dictionary.
        :param numeric_info_keywords: Info dictionary keywords corresponding to numeric episode-level metrics.
        :param collision_types: Possible collision types stored in the info dictionary.
        :param verbose: The verbosity level.
        :return: None.
        """
        super().__init__(verbose = verbose)
        self.rolling_window_size = rolling_window_size
        self.outcomes = outcomes
        self.numeric_info_keywords = numeric_info_keywords
        self.collision_types = collision_types

        self.rolling_outcomes = deque(maxlen = rolling_window_size)
        self.rolling_numeric_metrics = {key: deque(maxlen = rolling_window_size) for key in numeric_info_keywords}
        self.rolling_collision_types = deque(maxlen = rolling_window_size)

    def _on_step(self) -> bool:
        """
        Computes and logs rolling metrics via the Stable Baselines 3 logger.
        
        :return: Whether to continue training. Always returns True.
        """
        # Get the info dictionaries returned by each environment in the vectorized environment
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        # Get whether the episode is done for each environment in the vectorized environment
        dones = self.locals.get("dones", None)
        if dones is None:
            # Reconstruct dones from terminateds and truncateds
            terminateds = self.locals.get("terminateds", self.locals.get("terminated", None))
            truncateds = self.locals.get("truncateds", self.locals.get("truncated", None))
            if terminateds is None or truncateds is None:
                return True
            dones = [bool(terminated) or bool(truncated) for terminated, truncated in zip(terminateds, truncateds)]

        # Get the info dictionaries for finished episodes
        finished_episodes_info = [info for info, done in zip(infos, dones) if done and isinstance(info, dict)]

        # If no episodes finished, return True
        if not finished_episodes_info:
            return True

        # Store episode-level metrics in rolling windows
        for info in finished_episodes_info:
            outcome = info.get("outcome", None)
            if outcome is not None:
                self.rolling_outcomes.append(outcome)
            
            for info_keyword in self.numeric_info_keywords:
                value = info.get(info_keyword)
                if not isinstance(value, (int, float)):
                    continue
                self.rolling_numeric_metrics[info_keyword].append(float(value))
                # Log the mean of each episodic-level metric over the finished episodes
                self.logger.record_mean(f"episode/{info_keyword}", float(value))

            collision_type = info.get("collision_type")
            if collision_type is not None:
                self.rolling_collision_types.append(collision_type)

        # Compute rolling outcome rates
        if self.rolling_outcomes:
            counts = Counter(self.rolling_outcomes)
            total = len(self.rolling_outcomes)
            for outcome in self.outcomes:
                self.logger.record(f"outcomes/{outcome}_rate", counts.get(outcome, 0) / total)
        
        # Compute rolling means for numeric episode-level metrics
        for info_keyword, values in self.rolling_numeric_metrics.items():
            if values:
                self.logger.record(f"rolling/{info_keyword}_mean", sum(values) / len(values))

        # Compute rolling collision type rates
        if self.rolling_collision_types:
            counts = Counter(self.rolling_collision_types)
            total = len(self.rolling_collision_types)
            for collision_type in self.collision_types:
                self.logger.record(f"collision_type/{collision_type}_rate", counts.get(collision_type, 0) / total)

        return True
