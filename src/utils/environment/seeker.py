"""
This module provides the Seeker class for representing a seeker in the reactive avoidance environment.
"""
from minigrid.core.world_object import Ball


class Seeker(Ball):
    """
    A class for representing a seeker in the reactive avoidance environment.
    """
    def __init__(self) -> None:
        """
        Creates a Seeker object.
        """
        super().__init__("red")

    def can_overlap(self) -> bool:
        """
        Determines whether the agent can overlap with this seeker.
        
        :return: True.
        """
        return True
