import pygame
import numpy as np
from src.RL.environments.games.objects.board import Board


class SnakeGame:
    def __init__(self, args):
        self.board_size = args.board_size
        self.cell_size = args.pygame_cell_size
        self.display_size = args.pygame_display_size

        self.board = Board(
            board_size=self.board_size,
            background_value=args.background_value,
            background_color=args.background_color,
            grid_color=args.grid_color,
            border_value=args.border_value,
            border_color=args.border_color,
            snake_body_value=args.snake_body_value,
            snake_body_color=args.snake_body_color,
            snake_head_value=args.snake_head_value,
            snake_head_color=args.snake_head_color,
            snake_length=args.snake_init_length,
            food_value=args.food_value,
            food_color=args.food_color,
            num_foods=args.num_init_foods,
        )

        self.steps = 0
        self.scores = 0

        self.board_head_adds = [
            (0, -1),  # Left
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, 1)  # Right
        ]

        if args.use_pygame:
            pygame.init()
            self.display_surface = pygame.display.set_mode(size=(self.display_size[1], self.display_size[0]))
            pygame.display.set_caption(args.pygame_game_name)
            self.clock = pygame.time.Clock()

    def reset(self, num_foods, snake_length):
        self.board.reset(num_foods=num_foods, snake_length=snake_length)
        self.steps = 0
        self.scores = 0

    def display(self):
        self.board.draw(display_surface=self.display_surface, cell_size=self.cell_size, display_size=self.display_size)
        self.display_surface.blit(pygame.transform.rotate(surface=self.display_surface, angle=90), (0, 0))
        self.display_surface.blit(pygame.transform.flip(surface=self.display_surface, flip_x=False, flip_y=True), (0, 0))
        pygame.display.update()
        pygame.event.pump()

    def move(self, action):

        action = np.argmax(a=action, axis=0)
        h = self.board.snake.coords[0][0] + self.board_head_adds[action][0]
        w = self.board.snake.coords[0][1] + self.board_head_adds[action][1]
        self.board.snake.direction = action

        if (h, w) in self.board.border.coords:
            game_over = True
            self.board.snake.coords.pop(-1)
            self.board.snake.coords.insert(0, (h, w))

        elif (h, w) in self.board.snake.coords[3:-1]:
            game_over = True
            self.board.snake.coords.pop(-1)
            self.board.snake.coords.insert(0, (h, w))

        elif (h, w) in self.board.food.coords:

            self.board.snake.coords.insert(0, (h, w))

            if self.board.snake.get_length() == ((self.board_size[0] * self.board_size[1]) - len(self.board.border.coords)):
                game_over = True
            else:
                game_over = False

            self.scores += 1

            self.board.food.coords.remove((h, w))

            if self.board.get_empty_blocks() > 0:
                self.board.food.coords.append(self.board.food.get_one_food(
                    board_size=self.board_size,
                    invalid_coords=self.board.border.coords + self.board.snake.coords + self.board.food.coords)
                )

        else:
            game_over = False
            self.board.snake.coords.pop(-1)
            self.board.snake.coords.insert(0, (h, w))

        return game_over