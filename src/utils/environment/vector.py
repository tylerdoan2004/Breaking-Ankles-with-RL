"""
This module provides the Vector class for representing a vector in a two-dimensional gridworld.
"""
from dataclasses import dataclass


@dataclass(frozen = True)
class Vector:
    """
    A class for representing a vector in a two-dimensional gridworld.
    """
    x: int
    y: int

    def __add__(self, other: object) -> "Vector":
        """
        Implements the addition operator for Vector objects.

        :param other: The object to add.
        :return: A new Vector object representing the component-wise sum of the two vectors or NotImplemented.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __radd__(self, other: object) -> "Vector":
        """
        Implements the right addition operator for Vector objects.

        :param other: The object to add.
        :return: A new Vector object representing the component-wise sum of the two vectors or NotImplemented.
        """
        return self + other

    def __sub__(self, other: object) -> "Vector":
        """
        Implements the subtraction operator for Vector objects.
        
        :param other: The object to subtract.
        :return: A new Vector object representing the component-wise difference of this vector and the other vector or NotImplemented.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x - other.x, self.y - other.y)

    def __rsub__(self, other: object) -> "Vector":
        """
        Implements the right subtraction operator for Vector objects.
        
        :param other: The object to subtract.
        :return: A new Vector object representing the component-wise difference of the other vector and this vector or NotImplemented.
        """
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(other.x - self.x, other.y - self.y)
