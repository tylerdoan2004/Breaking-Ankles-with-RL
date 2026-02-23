"""
This module provides the Coordinates class for representing coordinates in a two-dimensional gridworld.
"""
from dataclasses import dataclass


@dataclass(frozen = True)
class Coordinates:
    """
    A class for representing coordinates in a two-dimensional gridworld.
    """
    x: int
    y: int

    @staticmethod
    def from_list(coordinates: list[int]) -> "Coordinates":
        """
        Creates a Coordinates object from a list of x- and y-coordinate components.

        :param coordinates: A list of x- and y-coordinate components.
        :return: A Coordinates object.
        """
        return Coordinates(coordinates[0], coordinates[1])
