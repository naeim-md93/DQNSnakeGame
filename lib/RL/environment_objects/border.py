import pygame
from typing import List, Tuple

import env


class Border:
    def __init__(self, empty_coords: set[Tuple[int, int]]):
        """
        Constructor of the Border class

        Parameters:
        ----------
        empty_coords: List[Tuple[int, int]]
            list of empty (h, w) coordinates in the board
        """
        self.value = env.BORDER_VALUE
        self.color = env.BORDER_COLOR
        self.coords = self.get_coords(empty_coords=empty_coords)

    def get_coords(self, empty_coords: set[Tuple[int, int]]) -> set[Tuple[int, int]]:
        """
        Get a list of border (h, w) coordinates

        Parameters:
        ----------
        empty_coords: List[Tuple[int, int]]
            list of empty (h, w) coordinates in the board

        Returns:
        ----------
        List[Tuple[int, int]]
            list of border (h, w) coordinates
        """
        min_height = min(empty_coords, key=lambda x: x[0])[0]
        max_height = max(empty_coords, key=lambda x: x[0])[0]
        min_width = min(empty_coords, key=lambda x: x[1])[1]
        max_width = max(empty_coords, key=lambda x: x[1])[1]

        coords = set()

        coords = coords.union({(i, min_width) for i in range(min_height, max_height + 1)})  # Left border
        coords = coords.union({(i, max_width) for i in range(min_height, max_height + 1)})  # Right border
        coords = coords.union({(min_height, i) for i in range(min_width + 1, max_width)})  # Up border
        coords = coords.union({(max_height, i) for i in range(min_width + 1, max_width)})  # Down border

        return coords

    def draw(self, display_surface: pygame.Surface) -> None:
        """
        Draw board object coordinates on the pygame display surface

        Parameters:
        ----------
        display_surface: pygame.Surface
            pygame display surface

        cell_size: int
            size of each cell in pixel
        """
        for coord in self.coords:
            top_pixel = coord[0] * env.CELL_SIZE + coord[0] + 1
            left_pixel = coord[1] * env.CELL_SIZE + coord[1] + 1
            border_rect = pygame.Rect(left_pixel, top_pixel, env.CELL_SIZE, env.CELL_SIZE)  # (left, top, width, height)
            pygame.draw.rect(surface=display_surface, color=self.color, rect=border_rect)
