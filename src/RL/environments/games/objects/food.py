import pygame
import numpy as np


class Food:
    def __init__(self, value, color, num_foods, board_size, invalid_coords):
        self.value = value
        self.color = color
        self.coords = self.get_coords(num_foods=num_foods, board_size=board_size, invalid_coords=invalid_coords)

    def get_coords(self, num_foods, board_size, invalid_coords):
        coords = []

        for _ in range(num_foods):
            coord = self.get_one_food(board_size=board_size, invalid_coords=invalid_coords + coords)
            coords.append(coord)

        return coords


    def get_one_food(self, board_size, invalid_coords):
        h = np.random.randint(low=0, high=board_size[0])
        w = np.random.randint(low=0, high=board_size[1])

        coord = (h, w)

        if coord in invalid_coords:
            coord = self.get_one_food(board_size=board_size, invalid_coords=invalid_coords)

        return coord


    def draw(self, display_surface, cell_size):
        for coord in self.coords:
            l = coord[0] * cell_size[1]
            t = coord[1] * cell_size[0]
            apple_rect = pygame.Rect(l, t, cell_size[1], cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(surface=display_surface, color=self.color, rect=apple_rect)
