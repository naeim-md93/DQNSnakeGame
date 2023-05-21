import random
import pygame

import env


class Snake:
    def __init__(self, init_length: int, empty_coords: set[tuple[int, int]]) -> None:
        """
        Constructor of the snake class
        :param init_length: initial length of the snake
        :param empty_coords: empty coordinates
        """
        self.direction = None
        self.body_value = env.SNAKE_BODY_VALUE
        self.body_color = env.SNAKE_BODY_COLOR
        self.head_value = env.SNAKE_HEAD_VALUE
        self.head_color = env.SNAKE_HEAD_COLOR
        self.actions = env.SNAKE_ACTIONS
        self.init_length = self.check_snake_length(empty_coords=empty_coords, length=init_length)

        self.coords = self.get_coords(empty_coords=empty_coords)

    def check_snake_length(self, empty_coords: set[tuple[int, int]], length: int) -> int:
        """
        Adjust snake length by empty coordinates
        :param empty_coords: empty coordinates in the board
        :param length: length of the snake
        :return: adjusted length of the snake
        """

        # If there is no empty room in the board
        if len(empty_coords) < 1:
            raise ValueError(f'{len(empty_coords)} available coords for creating snake!')

        # If length of snake is more than empty coordinates
        if length > len(empty_coords):
            length = len(empty_coords)

        return length

    def get_coords(self, empty_coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
        """
        Create snake coordinates
        :param empty_coords: empty coordinates
        :return: snake coordinates
        """
        coords = []

        # Add head coords
        coords.append(random.choice(seq=tuple(empty_coords)))

        # Add body coords
        for s in range(1, self.init_length):
            empty_coords.difference_update(set(coords))
            coords.append(
                self.get_next_body_coord(
                    prev_coord=coords[-1],
                    empty_coords=empty_coords,
                    segment=s
                )
            )

        return coords

    def get_next_body_coord(
            self,
            prev_coord: tuple[int, int],
            empty_coords: set[tuple[int, int]],
            segment: int
    ) -> tuple[int, int]:
        """
        Get next snake coordination based on the previous coordination
        :param prev_coord: previous snake coordination
        :param empty_coords: available coordinations
        :param segment: index of the snake body segment
        :return: (h, w) next snake coordinate
        """

        # Empty list of next possible coords
        next_coords = set()

        # Add all 4 next possible coords to list
        for n, c in env.BOARD_DIRECTIONS.items():
            next_coords.add(
                (
                    (
                         prev_coord[0] + c['ADD_TO_COORDS'][0],
                         prev_coord[1] + c['ADD_TO_COORDS'][1]
                    ), c['INVALID']
                )
            )

        # Remove coords that are invalid
        next_coords.intersection(empty_coords)

        # Select next coord randomly from next valid coords list
        nxc = random.choice(seq=tuple(next_coords))

        if segment == 1:
            coord, self.direction = nxc
        else:
            coord = nxc[0]

        return coord

    def get_length(self) -> int:
        """
        Get length of the snake
        :return: length of the snake
        """
        return len(self.coords)

    def draw(self, display_surface: pygame.Surface) -> None:
        """
        draw snake on pygame display surface
        :param display_surface: pygame display surface
        :param cell_size: size of each cell
        """
        for i, coord in enumerate(self.coords):

            top_pixel = coord[0] * env.CELL_SIZE + coord[0] + 1
            left_pixel = coord[1] * env.CELL_SIZE + coord[1] + 1

            snake_segment_rect = pygame.Rect(left_pixel, top_pixel, env.CELL_SIZE, env.CELL_SIZE)  # (left, top, width, height)
            pygame.draw.rect(
                surface=display_surface,
                color=self.head_color if i == 0 else self.body_color,
                rect=snake_segment_rect
            )
