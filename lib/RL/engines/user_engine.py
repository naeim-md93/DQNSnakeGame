import sys
import pygame
import numpy as np
from tqdm import tqdm

from ..environment import SnakeGame
import env


class UserEngine:
    def __init__(
            self,
            environment: SnakeGame,
            game_name: str,
            init_snake_length: int,
            snake_reduction_rate: int,
            init_num_foods: int,
            food_reduction_rate: int,
    ) -> None:
        self.environment = environment
        self.init_snake_length = init_snake_length
        self.snake_reduction_rate = snake_reduction_rate
        self.init_num_foods = init_num_foods
        self.food_reduction_rate = food_reduction_rate

        # Initialize pygame and pygame display
        pygame.init()
        self.display_surface = pygame.display.set_mode(size=(env.DISPLAY_SIZE[1], env.DISPLAY_SIZE[0]))
        pygame.display.set_caption(game_name)

        if env.DEBUG:
            print(f'{"#" * 10} Initializing User Engine {"#" * 10}')
            print(f'{self.init_snake_length=}')
            print(f'{self.snake_reduction_rate=}')
            print(f'{self.init_num_foods=}')
            print(f'{self.food_reduction_rate=}')
            print({"#" * 50})
            input('Initializing User Engine: ')

    def play_by_user(self, num_games: int) -> None:
        """
        Method for playing game by user
        """
        t = tqdm(iterable=range(1, num_games + 1))

        snake_length = self.init_snake_length
        num_foods = self.init_num_foods

        for n in t:
            # Initialize each game by setting game_over to False and reset the game
            game_over = False
            self.environment.reset(snake_length=snake_length, num_foods=num_foods)

            # Continue the game while it is not over
            while not game_over:
                # Display the board
                self.environment.display(display_surface=self.display_surface)

                # Get user action
                action = self.get_user_input(snake_direction=self.environment.board.snake.direction)

                # Play that action and get its game_over and reward state
                game_over, reward = self.environment.play_one_action(action=action)

                # Print it in the progress bar
                t.set_description_str(
                    desc=f'User Game: {n}, '
                         f'Step: {self.environment.steps}, '
                         f'Scores: {self.environment.scores}, '
                         f'Action: {action}, '
                         f'Reward: {reward:0.4f}, '
                         f'Game Over: {game_over}'
                )

            if self.snake_reduction_rate and (n % self.snake_reduction_rate == 0):
                if snake_length > env.MIN_SNAKE_LENGTH:
                    snake_length -= 1
                else:
                    snake_length = env.MIN_SNAKE_LENGTH

            if self.food_reduction_rate and (n % self.food_reduction_rate == 0):
                if num_foods > env.MIN_NUM_FOODS:
                    num_foods -= 1
                else:
                    num_foods = env.MIN_NUM_FOODS

    def get_user_input(self, snake_direction: int) -> np.ndarray:
        """
        Get user keyboard input and check its validation using snake_direction

        :param snake_direction: snake direction in [0, 1, 2, 3] for [left, up, right, down]
        :return: user action
        """
        done = False
        action = np.zeros(shape=(len(env.SNAKE_ACTIONS), ))

        # While the correct action is not performed by user
        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:

                    if (event.key == pygame.K_LEFT) and (snake_direction != env.BOARD_DIRECTIONS[env.LEFT]['INVALID']):
                        action[env.LEFT] = 1
                        done = True

                    if (event.key == pygame.K_UP) and (snake_direction != env.BOARD_DIRECTIONS[env.UP]['INVALID']):
                        action[env.UP] = 1
                        done = True

                    elif (event.key == pygame.K_RIGHT) and (snake_direction != env.BOARD_DIRECTIONS[env.RIGHT]['INVALID']):
                        action[env.RIGHT] = 1
                        done = True

                    elif (event.key == pygame.K_DOWN) and (snake_direction != env.BOARD_DIRECTIONS[env.DOWN]['INVALID']):
                        action[env.DOWN] = 1
                        done = True

        return action
