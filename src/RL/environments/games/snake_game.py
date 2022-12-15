import pygame
import numpy as np
from src.RL.environments.games.objects.snake import Snake
from src.RL.environments.games.objects.food import Food


class SnakeGame:
    def __init__(self, args):
        self.board_size = args.board_size
        self.background_color = args.background_color
        self.pygame_cell_size = args.pygame_cell_size
        self.pygame_display_size = args.pygame_display_size
        self.border_color = args.border_color
        self.grid_color = args.grid_color
        self.snake_head_color = args.snake_head_color
        self.snake_body_color = args.snake_body_color
        self.food_color = args.food_color

        self.steps = 0
        self.scores = 0
        self.game_over = False

        left_border = [(i, 0) for i in range(self.board_size[0])]
        right_border = [(i, self.board_size[1] - 1) for i in range(self.board_size[0])]
        up_border = [(0, i) for i in range(1, self.board_size[1] - 1)]
        down_border = [(self.board_size[0] - 1, i) for i in range(1, self.board_size[1] - 1)]
        self.borders = left_border + right_border + up_border + down_border

        self.board_head_adds = [
            (0, -1),  # Left
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, 1)  # Right
        ]

        if args.use_pygame:
            pygame.init()
            self.display_surf = pygame.display.set_mode(
                size=(args.pygame_display_size[1], args.pygame_display_size[0]))
            pygame.display.set_caption(args.pygame_game_name)
            self.clock = pygame.time.Clock()

    def reset(self, n_foods, snake_length):
        self.snake = Snake(
            board_size=self.board_size,
            invalid_coords=self.borders,
            snake_length=snake_length,
            head_color=self.snake_head_color,
            body_color=self.snake_body_color
        )
        self.food = Food(
            n_foods=n_foods,
            invalid_coords=self.borders+self.snake.coords,
            board_size=self.board_size,
            color=self.food_color
        )
        self.steps = 0
        self.scores = 0

    def draw_grid(self):
        for l in range(0, self.pygame_display_size[0], self.pygame_cell_size[0]):
            pygame.draw.line(surface=self.display_surf, color=self.grid_color, start_pos=(0, l), end_pos=(self.pygame_display_size[1], l))
        for t in range(0, self.pygame_display_size[1], self.pygame_cell_size[1]):
            pygame.draw.line(surface=self.display_surf, color=self.grid_color, start_pos=(t, 0), end_pos=(t, self.pygame_display_size[0]))

    def draw_borders(self, borders):
        for i in range(len(borders)):
            coord = borders[i]
            l = coord[0] * self.pygame_cell_size[1]
            t = coord[1] * self.pygame_cell_size[0]
            border_rect = pygame.Rect(l, t, self.pygame_cell_size[1], self.pygame_cell_size[0])  # (left, top, width, height)
            pygame.draw.rect(surface=self.display_surf, color=self.border_color, rect=border_rect)

    def display(self):
        self.display_surf.fill(color=self.background_color)
        self.food.draw(display_surf=self.display_surf, cell_size=self.pygame_cell_size)
        self.snake.draw(display_surf=self.display_surf, cell_size=self.pygame_cell_size)
        self.draw_borders(borders=self.borders)
        self.draw_grid()
        self.display_surf.blit(pygame.transform.rotate(surface=self.display_surf, angle=90), (0, 0))
        self.display_surf.blit(pygame.transform.flip(surface=self.display_surf, flip_x=False, flip_y=True), (0, 0))
        pygame.display.update()
        pygame.event.pump()

    def move(self, action):

        action = np.argmax(a=action, axis=0)
        h = self.snake.coords[0][0] + self.board_head_adds[action][0]
        w = self.snake.coords[0][1] + self.board_head_adds[action][1]
        self.snake.direction = action

        if (h, w) in self.borders:
            game_over = True
            self.snake.coords.pop(-1)
            self.snake.coords.insert(0, (h, w))

        elif (h, w) in self.snake.coords[:-1]:
            game_over = True
            self.snake.coords.pop(-1)
            self.snake.coords.insert(0, (h, w))

        elif (h, w) in self.food.coords:

            self.snake.coords.insert(0, (h, w))

            if self.snake.get_length() == ((self.board_size[0] - 2) * (self.board_size[1] - 2)):
                game_over = True
            else:
                game_over = False
            self.scores += 1

            self.food.coords.remove((h, w))
            self.food.coords.append(self.food.make(board_size=self.board_size, invalid_coords=self.borders + self.snake.coords + self.food.coords))

        else:
            game_over = False
            self.snake.coords.pop(-1)
            self.snake.coords.insert(0, (h, w))

        return game_over