import random
import pygame
import numpy as np
from PIL import Image

import env
from .agent_objects.reward import REWARDS
from .environment_objects.board import Board


class SnakeGame:
    def __init__(self) -> None:
        """
        Snake game object. Steps and Scores will be set to 0 once the game restarted
        """
        self.steps = None
        self.scores = None
        self.board = Board()

    def play_one_action(self, action: np.ndarray) -> [bool, float]:
        """
        Play one step by given action,
        :param action: action in form of [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        :return: Game over and reward state based on the action
        """

        self.steps += 1
        reward = 0
        action = np.argmax(a=action, axis=0)

        h = self.board.snake.coords[0][0] + self.board.directions[action]['ADD_TO_COORDS'][0]
        w = self.board.snake.coords[0][1] + self.board.directions[action]['ADD_TO_COORDS'][1]
        self.board.snake.direction = action

        if (h, w) in self.board.border.coords or (h, w) in self.board.snake.coords[:-1]:
            game_over = True
            reward += REWARDS['GAME_OVER']
            self.board.snake.coords.pop(-1)
            self.board.snake.coords.insert(0, (h, w))

        elif (h, w) in self.board.food.coords:
            self.scores += 1

            self.board.snake.coords.insert(0, (h, w))
            self.board.food.coords.remove((h, w))

            if len(self.board.get_coords(rm_border_coords=True, rm_snake_coords=True)):
                game_over = False

                empty_coords = self.board.get_coords(rm_all_coords=True)
                if len(empty_coords):
                    self.board.food.coords.add(random.choice(seq=tuple(empty_coords)))

            else:
                game_over = True

            reward += REWARDS['FOOD'](
                score=self.scores,
                max_score=len(self.board.get_coords(rm_border_coords=True)) - self.board.snake.init_length
            )

        elif self.steps == 100 * len(self.board.snake.coords):
            game_over = True
            reward += REWARDS['TIMEOUT']

        else:
            game_over = False
            reward += REWARDS['ELSE'](
                snake_coords=self.board.snake.coords,
                food_coords=self.board.food.coords,
                choice=(h, w),
                init_snake_length=self.board.snake.init_length
            )
            self.board.snake.coords.pop(-1)
            self.board.snake.coords.insert(0, (h, w))

        if env.DEBUG:
            print(f'{"#" * 10} PLAYING ONE STEP {"#" * 10}')
            print(f'{self.steps=}')
            print(f'{self.scores=}')
            print(f'{action=}')
            print(f'{(h, w)=}')
            print(f'{reward=}')
            print(f'{game_over=}')
            print(f'{vars(self.board.snake)=}')
            print(f'{vars(self.board.food)=}')
            input(f'{"#" * 10} PLAYING ONE STEP {"#" * 10}')

        return game_over, reward

    def get_step_info(self) -> [np.ndarray, np.ndarray, Image]:
        """
        Get step info. This includes:\n
        2d np array board,\n
        1d np array legal moves (if snake direction is right, if can not suddenly go left)
        Pillow image
        :return: Step info
        """
        state_2d = self.board.get_board()
        snake_coords = tuple(self.board.snake.coords)
        food_coords = tuple(self.board.food.coords)
        border_coords = tuple(self.board.border.coords)
        board_directions = self.board.directions
        snake_direction = self.board.snake.direction

        left = (
            snake_coords[0][0] + board_directions[env.LEFT]['ADD_TO_COORDS'][0],
            snake_coords[0][1] + board_directions[env.LEFT]['ADD_TO_COORDS'][1],
        )
        up = (
            snake_coords[0][0] + board_directions[env.UP]['ADD_TO_COORDS'][0],
            snake_coords[0][1] + board_directions[env.UP]['ADD_TO_COORDS'][1],
        )
        right = (
            snake_coords[0][0] + board_directions[env.RIGHT]['ADD_TO_COORDS'][0],
            snake_coords[0][1] + board_directions[env.RIGHT]['ADD_TO_COORDS'][1],
        )
        down = (
            snake_coords[0][0] + board_directions[env.DOWN]['ADD_TO_COORDS'][0],
            snake_coords[0][1] + board_directions[env.DOWN]['ADD_TO_COORDS'][1],
        )

        state_1d = np.array([
            (left in border_coords) or (left in snake_coords[1:]),
            (up in border_coords) or (up in snake_coords[1:]),
            (right in border_coords) or (right in snake_coords[1:]),
            (down in border_coords) or (down in snake_coords[1:]),
            snake_direction == env.LEFT,
            snake_direction == env.UP,
            snake_direction == env.RIGHT,
            snake_direction == env.DOWN,
            snake_coords[0][1] > food_coords[0][1],
            snake_coords[0][0] > food_coords[0][0],
            snake_coords[0][1] < food_coords[0][1],
            snake_coords[0][0] < food_coords[0][0],
        ], dtype=int)

        legal_moves = np.ones(shape=(len(self.board.snake.actions),))
        legal_moves[board_directions[snake_direction]['INVALID']] = 0

        img = Image.fromarray(obj=state_2d, mode='P')
        img.putpalette(
            data=self.board.color +
                 self.board.border.color +
                 self.board.snake.body_color +
                 self.board.snake.head_color +
                 self.board.food.color
        )
        img = np.array(img.convert(mode='RGB'))

        return state_2d, state_1d, legal_moves, img

    def reset(self, num_foods: int, snake_length: int) -> None:
        """
        Reset the game by resetting the board and set steps and scores to 0
        :param num_foods: number of foods to be created in the board
        :param snake_length: length of snake to be initialized
        :return: None
        """
        self.board.reset(num_foods=num_foods, snake_length=snake_length)
        self.steps = 0
        self.scores = 0

    def display(self, display_surface: pygame.display) -> None:
        """
        Display game on pygame display surface
        :param display_surface: pygame display surface
        :return: None
        """
        self.board.draw(display_surface=display_surface)
        pygame.display.update()
        pygame.event.pump()
