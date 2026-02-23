"""
This module provides the Coordinates class for representing coordinates in a two-dimensional gridworld.
"""
from dataclasses import dataclass
from types import NotImplementedType
from typing import overload
from utils.environment.vector import Vector

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

    def __add__(self, other: object) -> "Coordinates":
        """
        Implements the addition operator for Coordinates objects.

        :param other: The object to add.
        :return: A new Coordinates object representing the component-wise sum of the coordinates with the vector or NotImplemented.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return Coordinates(self.x + other.x, self.y + other.y)

    def __radd__(self, other: object) -> "Coordinates":
        """
        Implements the right addition operator for Coordinates objects.

        :param other: The object to add.
        :return: A new Coordinates object representing the component-wise sum of the coordinates with the vector or NotImplemented.
        """
        return self + other

    @overload
    def __sub__(self, other: Vector) -> "Coordinates":
        ...

    @overload
    def __sub__(self, other: "Coordinates") -> Vector:
        ...

    def __sub__(self, other: object) -> "Coordinates | Vector":
        """
        Implements the subtraction operator for Coordinates objects.
        
        :param other: The object to subtract.
        :return: A new Coordinates object representing the component-wise difference of the coordinates with the vector, a Vector object representing the component-wise difference of the two coordinates, or NotImplemented.
        """
        if isinstance(other, Vector):
            return Coordinates(self.x - other.x, self.y - other.y)
        if isinstance(other, Coordinates):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented

    def __rsub__(self, other: object) -> NotImplementedType:
        """
        Implements the right subtraction operator for Coordinates objects.
        
        :param other: The object to subtract.
        :return: NotImplemented.
        """
        return NotImplemented
