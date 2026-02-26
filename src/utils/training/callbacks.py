"""
This module provides callbacks for logging an agent's training progress.
"""
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
    def __init__(self, *, video_directory: str, video_name_prefix: str, environment_factory: VideoRecordableEnvironmentFactory, seed: int, recording_frequency: int, verbose: int = 0):
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
                    video_name_prefix = f"{self.video_name_prefix}-timestep-{self.next_recording_timestep}",
                    environment_factory = self.environment_factory,
                    agent = StableBaselines3Agent(self.model),
                    seed = self.seed
                )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[VideoCallback] Failed to record video at timestep {self.next_recording_timestep}: {e}")
            finally:
                self.next_recording_timestep += self.recording_frequency
