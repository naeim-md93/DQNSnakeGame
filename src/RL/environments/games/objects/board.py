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
            background_value,
            background_color,
            grid_color,
            border_value,
            border_color,
            snake_body_value,
            snake_body_color,
            snake_head_value,
            snake_head_color,
            snake_length,
            food_value,
            food_color,
            num_foods,
    ):
        self.board_head_adds = [
            (0, -1),  # Left
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, 1)  # Right
        ]
        self.board_size = board_size
        self.background_value = background_value
        self.background_color = background_color

        self.border = Border(board_size=self.board_size, value=border_value, color=border_color)
        self.grid = Grid(color=grid_color)
        self.snake = Snake(
            board_size=self.board_size,
            body_value=snake_body_value,
            body_color=snake_body_color,
            head_value=snake_head_value,
            head_color=snake_head_color,
            length=self.check_snake_length(length=snake_length),
            invalid_coords=self.border.coords,
        )

        self.food = Food(
            value=food_value,
            color=food_color,
            num_foods=self.check_num_foods(num_foods=num_foods),
            board_size=self.board_size,
            invalid_coords=self.border.coords + self.snake.coords
        )


    def __repr__(self):
        board = np.ones(shape=self.board_size, dtype=np.uint8) * self.background_value

        for c in self.border.coords:
            board[c[0], c[1]] = self.border.value

        board[self.snake.coords[0][0], self.snake.coords[0][1]] = self.snake.head_value

        for c in self.snake.coords[1:]:
            board[c[0], c[1]] = self.snake.body_value

        for c in self.food.coords:
            board[c[0], c[1]] = self.food.value

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

    def get_empty_blocks(self):
        return (self.board_size[0] * self.board_size[1]) - len(self.border.coords) - len(self.snake.coords) - len(self.food.coords)

    def reset(self, snake_length, num_foods):
        self.border = Border(board_size=self.board_size, value=self.border.value, color=self.border.color)
        self.grid = Grid(color=self.grid.color)
        self.snake = Snake(
            board_size=self.board_size,
            body_value=self.snake.body_value,
            body_color=self.snake.body_color,
            head_value=self.snake.head_value,
            head_color=self.snake.head_color,
            length=self.check_snake_length(length=snake_length),
            invalid_coords=self.border.coords,
        )

        self.food = Food(
            value=self.food.value,
            color=self.food.color,
            num_foods=self.check_num_foods(num_foods=num_foods),
            board_size=self.board_size,
            invalid_coords=self.border.coords + self.snake.coords
        )

    def draw(self, display_surface, cell_size, display_size):
        display_surface.fill(color=self.background_color)
        self.food.draw(display_surface=display_surface, cell_size=cell_size)
        self.snake.draw(display_surface=display_surface, cell_size=cell_size)
        self.border.draw(display_surface=display_surface, cell_size=cell_size)
        self.grid.draw(display_surface=display_surface, cell_size=cell_size, display_size=display_size)

if __name__ == '__main__':
    WHITE = [255, 255, 255]  # Background
    DARK_GREEN = [0, 155, 0]  # Snake Body
    GREEN = [0, 255, 0]  # Snake Head
    RED = [255, 0, 0]  # Apple
    BLACK = [0, 0, 0]  # Borders
    DARK_GRAY = (100, 100, 100)  # Grid

    board = Board(
        board_size=[14, 14],
        background_value=0,
        background_color=WHITE,
        grid_color=DARK_GRAY,
        border_value=1,
        border_color=BLACK,
        snake_body_value=2,
        snake_body_color=GREEN,
        snake_head_value=3,
        snake_head_color=DARK_GREEN,
        snake_length=4,
        food_value=4,
        food_color=RED,
        num_foods=120
    )
    print(board)

    board.reset(snake_length=4, num_foods=4)
    print(board)
