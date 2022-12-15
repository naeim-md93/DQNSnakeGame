import pygame
import numpy as np


class Food:
    def __init__(self, n_foods, invalid_coords, board_size, color):
        self.color = color
        self.coords = self.get(n_foods=n_foods, invalid_coords=invalid_coords, board_size=board_size)

    def make(self, board_size, invalid_coords):
        h = np.random.randint(low=1, high=board_size[0] - 1)
        w = np.random.randint(low=1, high=board_size[1] - 1)

        c = (h, w)

        if c in invalid_coords:
            c = self.make(board_size=board_size, invalid_coords=invalid_coords)
        return c

    def get(self, n_foods, invalid_coords, board_size):

        coords = []

        for i in range(n_foods):
            c = self.make(board_size=board_size, invalid_coords=invalid_coords + coords)
            coords.append(c)
        return coords

    def draw(self, display_surf, cell_size):
        for i in range(len(self.coords)):
            coord = self.coords[i]
            l = coord[0] * cell_size[1]
            t = coord[1] * cell_size[0]
            apple_rect = pygame.Rect(l, t, cell_size[1], cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(surface=display_surf, color=self.color, rect=apple_rect)
