"""
A module for statically type-checking environments.
"""
from typing import Any, Literal, Protocol, TypeVar
from gymnasium import Env


EnvTypeCovariant = TypeVar("EnvTypeCovariant", bound = Env, covariant = True)
class VideoRecordableEnvironmentFactory(Protocol[EnvTypeCovariant]):
    """
    A protocol for creating video recordable environments.
    """
    def __call__(self, *args: Any, render_mode: Literal["rgb_array"], **kwargs: Any) -> EnvTypeCovariant:
        """
        Creates a video recordable environment.
        
        :param render_mode: The render mode to use for the environment.
        :return: A video recordable environment.
        """
        ...
