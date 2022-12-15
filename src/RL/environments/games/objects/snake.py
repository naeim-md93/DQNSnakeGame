import pygame
import numpy as np


class Snake:
    def __init__(self, board_size, invalid_coords, snake_length, head_color, body_color):
        self.direction = np.random.randint(low=0, high=4)
        self.head_color = head_color
        self.body_color = body_color
        self.coords = self.get(
            snake_length=snake_length,
            invalid_coords=invalid_coords,
            board_size=board_size
        )

    def get_length(self):
        return len(self.coords)

    def get(self, board_size, invalid_coords, snake_length):
        coords = []

        if self.direction == 0:
            h = np.random.randint(low=1, high=board_size[0] - 1)
            w = np.random.randint(low=1, high=board_size[1] - snake_length)
            for i in range(snake_length):
                coords.insert(i, (h, w + i))

        elif self.direction == 1:
            h = np.random.randint(low=1, high=board_size[0] - snake_length)
            w = np.random.randint(low=1, high=board_size[1] - 1)
            for i in range(snake_length):
                coords.insert(i, (h + i, w))
        elif self.direction == 2:
            h = np.random.randint(low=snake_length, high=board_size[0] - 1)
            w = np.random.randint(low=1, high=board_size[1] - 1)
            for i in range(snake_length):
                coords.insert(i, (h - i, w))
        else:
            h = np.random.randint(low=1, high=board_size[0] - 1)
            w = np.random.randint(low=snake_length, high=board_size[1] - 1)
            for i in range(snake_length):
                coords.insert(i, (h, w - i))

        flag = False

        for c in coords:
            if c in invalid_coords:
                flag = True
        if flag:
            coords = self.get(board_size=board_size, invalid_coords=invalid_coords, snake_length=snake_length)

        return coords

    def draw(self, display_surf, cell_size):
        for i in range(self.get_length()):
            coord = self.coords[i]

            l = coord[0] * cell_size[1]
            t = coord[1] * cell_size[0]

            snake_segment_rect = pygame.Rect(l, t, cell_size[1], cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(
                surface=display_surf,
                color=self.head_color if i == 0 else self.body_color,
                rect=snake_segment_rect
            )