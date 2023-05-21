import pygame
import numpy as np

from .border import Border
from .grid import Grid
from .snake import Snake
from .food import Food

import env


class Board:
    def __init__(self) -> None:
        """
        Creates a board with its objects
        """
        self.size = env.BOARD_SIZE
        self.color = env.BACKGROUND_COLOR
        self.value = env.BACKGROUND_VALUE
        self.directions = env.BOARD_DIRECTIONS

        self.grid = None
        self.border = None
        self.snake = None
        self.food = None

    def get_coords(
            self,
            rm_border_coords: bool = False,
            rm_snake_coords: bool = False,
            rm_food_coords: bool = False,
            rm_all_coords: bool = False
    ) -> set[tuple[int, int]]:
        """
        Get all (h, w) coordinates in the board that are empty
        :return: List of all (h, w) coordinates in the board
        """

        coords = {(h, w) for h in range(self.size[0]) for w in range(self.size[1])}

        if rm_all_coords or rm_border_coords:
            coords.difference_update(self.border.coords)

        if rm_all_coords or rm_snake_coords:
            coords.difference_update(set(self.snake.coords))

        if rm_all_coords or rm_food_coords:
            coords.difference_update(self.food.coords)

        return coords

    def get_board(self) -> np.ndarray:
        """
        Get 2d np array of the board using object values
        :return: 2d np array of the board
        """

        board = np.ones(shape=self.size, dtype=np.uint8) * self.value

        for c in self.border.coords:
            board[c[0], c[1]] = self.border.value

        board[self.snake.coords[0][0], self.snake.coords[0][1]] = self.snake.head_value

        for c in self.snake.coords[1:]:
            board[c[0], c[1]] = self.snake.body_value

        for c in self.food.coords:
            board[c[0], c[1]] = self.food.value

        return board

    def __repr__(self):
        return np.array2string(a=self.get_board())

    def reset(self, snake_length: int, num_foods: int) -> None:
        """
        Restart board with its objects
        :param num_foods: number of foods to be created in the board
        :param snake_length: initial length of the snake
        """
        self.grid = Grid()
        self.border = Border(empty_coords=self.get_coords())
        self.snake = Snake(init_length=snake_length, empty_coords=self.get_coords(rm_border_coords=True))
        self.food = Food(num_foods=num_foods, empty_coords=self.get_coords(rm_border_coords=True, rm_snake_coords=True))

    def draw(self, display_surface: pygame.Surface) -> None:
        """
        Draw board on pygame display surface
        :param display_surface: pygame display surface
        :param display_size: display size
        """
        display_surface.fill(color=self.color)
        self.grid.draw(display_surface=display_surface)
        self.border.draw(display_surface=display_surface)
        self.snake.draw(display_surface=display_surface)
        self.food.draw(display_surface=display_surface)
