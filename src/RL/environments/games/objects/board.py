import pygame
import numpy as np
from src.RL.environments.games.objects.border import Border
from src.RL.environments.games.objects.grid import Grid
from src.RL.environments.games.objects.snake import Snake
from src.RL.environments.games.objects.food import Food


class Board:
    def __init__(
            self,
            board_size,
            board_directions,
            background_value,
            background_color,
            grid_color,
            border_value,
            border_color,
            snake_length,
            snake_actions,
            snake_body_value,
            snake_body_color,
            snake_head_value,
            snake_head_color,
            num_foods,
            food_value,
            food_color,
    ):
        self.board_size = board_size
        self.board_directions = board_directions
        self.background_value = background_value
        self.background_color = background_color

        self.grid = Grid(color=grid_color)
        self.border = Border(board_size=self.board_size, value=border_value, color=border_color)

        self.snake = Snake(
            board_size=self.board_size,
            board_directions=self.board_directions,
            length=self.check_snake_length(length=snake_length),
            actions=snake_actions,
            body_value=snake_body_value,
            body_color=snake_body_color,
            head_value=snake_head_value,
            head_color=snake_head_color,
            invalid_coords=self.border.coords,
        )

        self.food = Food(
            board_size=self.board_size,
            num_foods=self.check_num_foods(num_foods=num_foods),
            value=food_value,
            color=food_color,
            invalid_coords=self.border.coords + self.snake.coords
        )

    def __repr__(self):
        board = self.get_board()
        return np.array2string(a=board)

    def check_snake_length(self, length):
        max_length = (self.board_size[0] * self.board_size[1]) - len(self.border.coords) - 1
        if length > max_length:
            length = max_length
        return length

    def check_num_foods(self, num_foods):
        max_foods = (self.board_size[0] * self.board_size[1]) - len(self.border.coords) - len(self.snake.coords)
        if num_foods > max_foods:
            num_foods = max_foods
        return num_foods

    def get_board(self):
        board = np.ones(shape=self.board_size, dtype=np.uint8) * self.background_value

        for c in self.border.coords:
            board[c[0], c[1]] = self.border.value

        board[self.snake.coords[0][0], self.snake.coords[0][1]] = self.snake.head_value

        for c in self.snake.coords[1:]:
            board[c[0], c[1]] = self.snake.body_value

        for c in self.food.coords:
            board[c[0], c[1]] = self.food.value

        return board

    def get_empty_blocks(self):
        empty_blocks = self.board_size[0] * self.board_size[1]
        empty_blocks = empty_blocks - len(self.border.coords)
        empty_blocks = empty_blocks - len(self.snake.coords)
        empty_blocks = empty_blocks - len(self.food.coords)
        return empty_blocks

    def reset(self, snake_length, num_foods):
        self.border = Border(board_size=self.board_size, value=self.border.value, color=self.border.color)
        self.grid = Grid(color=self.grid.color)
        self.snake = Snake(
            board_size=self.board_size,
            board_directions=self.board_directions,
            length=self.check_snake_length(length=snake_length),
            actions=self.snake.actions,
            body_value=self.snake.body_value,
            body_color=self.snake.body_color,
            head_value=self.snake.head_value,
            head_color=self.snake.head_color,
            invalid_coords=self.border.coords,
        )

        self.food = Food(
            board_size=self.board_size,
            num_foods=self.check_num_foods(num_foods=num_foods),
            value=self.food.value,
            color=self.food.color,
            invalid_coords=self.border.coords + self.snake.coords
        )

    def draw(self, display_surface, cell_size, display_size):
        display_surface.fill(color=self.background_color)
        self.food.draw(display_surface=display_surface, cell_size=cell_size)
        self.snake.draw(display_surface=display_surface, cell_size=cell_size)
        self.border.draw(display_surface=display_surface, cell_size=cell_size)
        self.grid.draw(display_surface=display_surface, cell_size=cell_size, display_size=display_size)
