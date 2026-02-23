"""
This module provides the GridDimensions class for representing the dimensions of a two-dimensional gridworld.
"""
from dataclasses import dataclass
from utils.environment.coordinates import Coordinates


@dataclass(frozen = True)
class GridDimensions:
    """
    A class for representing the dimensions of a two-dimensional gridworld.
    """
    width: int
    height: int

    @staticmethod
    def from_tuple(grid_dimensions: tuple[int, int]) -> "GridDimensions":
        """
        Creates a GridDimensions object from a tuple of width and height.
        
        :param grid_dimensions: A tuple of width and height.
        :return: A GridDimensions object.
        """
        return GridDimensions(grid_dimensions[0], grid_dimensions[1])

    def contains_coordinates(self, coordinates: Coordinates) -> bool:
        """
        Checks if the coordinates are within the grid dimensions.
        
        :param coordinates: The coordinates to check.
        :return: True if the coordinates are within the grid dimensions, False otherwise.
        """
        return 0 <= coordinates.x < self.width and 0 <= coordinates.y < self.height
