import pygame
import numpy as np


class Snake:
    def __init__(
            self,
            board_size,
            body_value,
            body_color,
            head_value,
            head_color,
            length,
            invalid_coords,
    ):
        self.direction = np.random.randint(low=0, high=4)
        self.body_value = body_value
        self.body_color = body_color
        self.head_value = head_value
        self.head_color = head_color
        self.board_tail_adds = [
            (0, 1),  # direction Left -> add to right
            (1, 0),  # direction up -> add to down
            (-1, 0),  # direction down -> add to up
            (0, -1),  # direction right -> add to left
        ]

        self.coords = self.get_coords(board_size=board_size, invalid_coords=invalid_coords, length=length, )

    def get_coords(self, board_size, invalid_coords, length):
        coords = []

        coords.append(self.get_head_coords(board_size=board_size, invalid_coords=invalid_coords))

        for _ in range(1, length):
            coord = self.get_next_body_coord(prev_coord=coords[-1], invalid_coords=invalid_coords + coords)
            coords.append(coord)

        return coords

    def get_length(self):
        return len(self.coords)

    def get_head_coords(self, board_size, invalid_coords):

        h = np.random.randint(low=0, high=board_size[0])
        w = np.random.randint(low=0, high=board_size[1])

        head_coords = (h, w)

        if head_coords in invalid_coords:
            head_coords = self.get_head_coords(board_size=board_size, invalid_coords=invalid_coords)

        return head_coords

    def get_next_body_coord(self, prev_coord, invalid_coords):
        next_coords = []

        for i, c in enumerate(self.board_tail_adds):
            next_coords.append((prev_coord[0] + c[0], prev_coord[1] + c[1]))

        next_coords = [c for c in next_coords if c not in invalid_coords]

        coord = next_coords[np.random.randint(low=0, high=len(next_coords))]

        return coord

    def draw(self, display_surface, cell_size):
        for i, coord in enumerate(self.coords):
            l = coord[0] * cell_size[1]
            t = coord[1] * cell_size[0]

            snake_segment_rect = pygame.Rect(l, t, cell_size[1], cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(
                surface=display_surface,
                color=self.head_color if i == 0 else self.body_color,
                rect=snake_segment_rect
            )