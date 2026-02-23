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

    def __add__(self, other: "Coordinates") -> "Coordinates":
        """
        Adds this Coordinates object to another Coordinates object component-wise.
        
        :param other: The Coordinates object to add.
        :return: A new Coordinates object representing the component-wise sum of the two Coordinates objects.
        """
        if not isinstance(other, Coordinates):
            return NotImplemented
        return Coordinates(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Coordinates") -> "Coordinates":
        """
        Subtracts another Coordinates object from this Coordinates object component-wise.
        
        :param other: The Coordinates object to subtract.
        :return: A new Coordinates object representing the component-wise difference between the two Coordinates objects.
        """
        if not isinstance(other, Coordinates):
            return NotImplemented
        return Coordinates(self.x - other.x, self.y - other.y)
