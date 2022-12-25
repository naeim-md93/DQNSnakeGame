import pygame
import numpy as np


class Snake:
    def __init__(
            self,
            board_size,
            board_directions,
            length,
            actions,
            body_value,
            body_color,
            head_value,
            head_color,
            invalid_coords,
    ):

        self.actions = actions
        self.direction = None
        self.body_value = body_value
        self.body_color = body_color
        self.head_value = head_value
        self.head_color = head_color
        self.coords = self.get_coords(
            board_size=board_size,
            invalid_coords=invalid_coords,
            length=length,
            board_directions=board_directions
        )

    def get_length(self):
        return len(self.coords)

    def get_coords(self, board_size, invalid_coords, length, board_directions):
        coords = []

        # Add head coords
        coords.append(self.get_head_coords(board_size=board_size, invalid_coords=invalid_coords))

        # Add body coords
        for s in range(1, length):
            coord = self.get_next_body_coord(
                board_directions=board_directions,
                prev_coord=coords[-1],
                invalid_coords=invalid_coords + coords,
                segment=s
            )
            coords.append(coord)

        return coords

    def get_head_coords(self, board_size, invalid_coords):

        h = np.random.randint(low=0, high=board_size[0])
        w = np.random.randint(low=0, high=board_size[1])

        head_coords = (h, w)

        if head_coords in invalid_coords:
            head_coords = self.get_head_coords(board_size=board_size, invalid_coords=invalid_coords)

        return head_coords

    def get_next_body_coord(self, board_directions, prev_coord, invalid_coords, segment):

        # Empty list of next possible coords
        next_coords = []

        # Add all 4 next possible coords to list
        for n, c in board_directions.items():
            next_coords.append(
                (
                    (prev_coord[0] + c['ADD_TO_COORDS'][0], prev_coord[1] + c['ADD_TO_COORDS'][1]),
                    c['INVALID']
                )
            )

        # Remove coords that are invalid
        next_coords = [c for c in next_coords if c[0] not in invalid_coords]

        # Select next coord randomly from next valid coords list
        nxc = next_coords[np.random.randint(low=0, high=len(next_coords))]

        if segment == 1:
            coord, self.direction = nxc
        else:
            coord = nxc[0]

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
