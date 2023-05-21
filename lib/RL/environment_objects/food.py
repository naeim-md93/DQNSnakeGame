import random
import pygame

import env


class Food:
    def __init__(self, num_foods: int, empty_coords: set[tuple[int, int]]) -> None:
        """
        Constructor of the food class
        :param num_foods: number of foods to be created
        :param empty_coords: empty coordinates
        """
        self.value = env.FOOD_VALUE
        self.color = env.FOOD_COLOR
        self.init_foods = self.check_num_foods(empty_coords=empty_coords, num_foods=num_foods)
        self.coords = self.get_coords(empty_coords=empty_coords)

    def check_num_foods(self, empty_coords: set[tuple[int, int]], num_foods: int) -> int:
        """
        Check if number of foods are less or equal valid coords
        :param empty_coords: list of valid/empty (h, w) coordinates
        :param num_foods: number of foods to be created
        :return: adjusted number of foods
        """

        # If there is no empty coords
        if len(empty_coords) < 1:
            raise ValueError(f'{len(empty_coords)} available coords for creating food!')

        # If number of foods is greater than empty coords
        if num_foods > len(empty_coords):
            num_foods = len(empty_coords)

        return num_foods

    def get_coords(self, empty_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Create food coordinates
        :param num_foods: number of foods to be created
        :param empty_coords: empty coordinates
        :return: food coordinates
        """
        coords = set()

        # add coord
        for _ in range(self.init_foods):
            empty_coords.difference_update(coords)
            coords.add(random.choice(seq=tuple(empty_coords)))

        return coords

    def draw(self, display_surface: pygame.surface) -> None:
        """
        draw foods on pygame display surface
        :param display_surface: pygame display surface
        :param cell_size: size of each cell
        :return: None
        """
        for coord in self.coords:
            top_pixel = coord[0] * env.CELL_SIZE + coord[0] + 1
            left_pixel = coord[1] * env.CELL_SIZE + coord[1] + 1
            apple_rect = pygame.Rect(left_pixel, top_pixel, env.CELL_SIZE, env.CELL_SIZE)  # (left, top, width, height)
            pygame.draw.rect(surface=display_surface, color=self.color, rect=apple_rect)
