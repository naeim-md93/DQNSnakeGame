import pygame
import numpy as np
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter

import env
from ..environment import SnakeGame
from ..agent import QAgent


class AgentEngine:
    def __init__(
            self,
            environment: SnakeGame,
            agent: QAgent,
            init_snake_length: int,
            snake_reduction_rate: int,
            init_num_foods: int,
            food_reduction_rate: int,
            logs_path: str,
            game_name: str
    ):
        self.environment = environment
        self.agent = agent
        self.pygame = pygame
        self.init_snake_length = init_snake_length
        self.snake_reduction_rate = snake_reduction_rate
        self.init_num_foods = init_num_foods
        self.food_reduction_rate = food_reduction_rate
        self.logger = SummaryWriter(log_dir=logs_path)

        if env.PYGAME:
            pygame.init()
            self.display_surface = pygame.display.set_mode(size=(env.DISPLAY_SIZE[1], env.DISPLAY_SIZE[0]))
            pygame.display.set_caption(game_name)

        if env.DEBUG:
            print(f'{"#" * 10} Initializing Agent Engine {"#" * 10}')
            print(f'{self.init_snake_length=}')
            print(f'{self.snake_reduction_rate=}')
            print(f'{self.init_num_foods=}')
            print(f'{self.food_reduction_rate=}')
            print(f'{vars(self.logger)=}')
            print({"#" * 50})
            input('Initializing Agent Engine: ')

    def play_by_agent(self, num_games: int) -> None:
        """
        Method for playing game by agent
        """
        t = tqdm(iterable=range(1, num_games + 1))

        # Initialize epsilon for exploration/exploitation trade-off
        # Initialize eta for important/un-important trade-ff
        # Initialize snake length for reduction rate
        # Initialize num foods for reduction rate
        epsilon = env.INIT_EPSILON
        eta = env.INIT_ETA
        snake_length = self.init_snake_length
        num_foods = self.init_num_foods

        for n in t:
            self.play_one_game(n=n, epsilon=epsilon, eta=eta, snake_length=snake_length, num_foods=num_foods)

            # Reduce epsilon using epsilon-greedy strategy
            if n <= env.EXPLORE_GAMES:
                epsilon = epsilon + (env.FINAL_EPSILON - env.INIT_EPSILON) / env.EXPLORE_GAMES

            # Reduce eta using epsilon-greedy strategy
            if env.EXPLORE_GAMES < n <= env.TRAIN_GAMES:
                eta = eta + (env.FINAL_ETA - env.INIT_ETA) / (env.TRAIN_GAMES - env.EXPLORE_GAMES)

            # Reduce snake length using snake reduction rate
            if self.snake_reduction_rate and (n % self.snake_reduction_rate == 0):
                if snake_length > env.MIN_SNAKE_LENGTH:
                    snake_length -= 1
                else:
                    snake_length = env.MIN_SNAKE_LENGTH

            # Reduce num foods using food reduction rate
            if self.food_reduction_rate and (n % self.food_reduction_rate == 0):
                if num_foods > env.MIN_NUM_FOODS:
                    num_foods -= 1
                else:
                    num_foods = env.MIN_NUM_FOODS

    def play_one_game(self, n, epsilon, eta, snake_length, num_foods):
        """
        Initialize each game by:
            setting game_over to False,
            resetting rewards, short memory losses, and game images containers
        """
        game_over = False
        rewards_total = [0]
        short_memory_loss_total = [0]
        short_memory_loss_left = [0]
        short_memory_loss_up = [0]
        short_memory_loss_right= [0]
        short_memory_loss_down = [0]
        lm_losses = [0, 0, 0, 0]
        game_images = []
        state2ds = deque(maxlen=10)
        state1ds = deque(maxlen=10)

        if env.DEBUG:
            print(f'{"#" * 10} Play One GAME {"#" * 10}')
            print(f'{n=}')
            print(f'{epsilon=}')
            print(f'{eta=}')
            print(f'{snake_length=}')
            print(f'{num_foods=}')

        # reset the game
        self.environment.reset(snake_length=snake_length, num_foods=num_foods)

        # Then get the initial environment info
        state2d0, state1d0, legal_moves0, img = self.environment.get_step_info()  # (14, 14), (4, )
        state2ds.append(state2d0)
        state1ds.append(state1d0)
        game_images.append(img)

        # Display using pygame
        if env.PYGAME:
            self.environment.display(display_surface=self.display_surface)

        if env.DEBUG:
            print(f'{"#" * 10} GET STATE 0 INFO {"#" * 10}')
            print(f'State2d0:\n{np.array2string(state2d0)}')
            print(f'State1d0:\n{state1d0}')
            print(f'legal_moves0: {legal_moves0=}')
            input(f'{"#" * 10} GET STATE 0 INFO  {"#" * 10}')

        # Continue the game while it is not over
        while not game_over:

            # Get an action from agent
            action, random_value = self.agent.get_agent_action(
                state2ds=state2ds,
                state1ds=state1ds,
                legal_moves=legal_moves0,
                epsilon=epsilon
            )

            # Play that action and get its game_over and reward state
            game_over, reward = self.environment.play_one_action(action=action)
            rewards_total.append(reward)

            # Then get the new environment info
            state2d1, state1d1, legal_moves1, img = self.environment.get_step_info()  # (14, 14), (4, )
            game_images.append(img)

            # Display using pygame
            if env.PYGAME:
                self.environment.display(display_surface=self.display_surface)

            if env.DEBUG:
                print(f'{"#" * 10} GET STATE 1 INFO {"#" * 10}')
                print(f'State2d1:\n{np.array2string(state2d1)}')
                print(f'State1d1:\n{state1d1}')
                print(f'legal_moves1: {legal_moves1=}')
                input(f'{"#" * 10} GET STATE 1 INFO  {"#" * 10}')

            xp = (
                np.stack(arrays=state2ds, axis=0),  # State0: (t, 14, 14)
                np.stack(arrays=state1ds, axis=0),  # State0: (t, 12)
                np.array([legal_moves0]),  # legal_moves0: (1, 4)
                np.array([action]),  # action: (1, 4)
                np.array([[reward]]),  # reward: (1, 1)
                np.array([[game_over]]),  # game_over: (1, 1)
                np.stack(arrays=list(state2ds) + [state2d1], axis=0),  # State1: (t + 1, 14, 14)
                np.stack(arrays=list(state1ds) + [state1d1], axis=0),  # State1: (t + 1, 12)
                np.array([legal_moves1]),  # legal_moves0: (1, 4)
            )

            # Store this experience in memory pool
            sizes = self.agent.remember(xp, reward)

            if random_value >= epsilon:
                sm_losses = self.agent.train_short_memory(xp=xp)
                short_memory_loss_left.append(sm_losses[env.LEFT])
                short_memory_loss_up.append(sm_losses[env.UP])
                short_memory_loss_right.append(sm_losses[env.RIGHT])
                short_memory_loss_down.append(sm_losses[env.DOWN])
                short_memory_loss_total.append(sm_losses.sum())

            state2ds.append(state2d1)
            state1ds.append(state1d1)
            legal_moves0 = legal_moves1

        if n > env.EXPLORE_GAMES:
            lm_losses = self.agent.train_long_memory(eta=eta)

        game_images = np.stack(arrays=game_images, axis=0)
        self.logger.add_scalar(tag='Epsilon', scalar_value=epsilon, global_step=n)
        self.logger.add_scalar(tag='Eta', scalar_value=eta, global_step=n)
        self.logger.add_scalar(tag='Scores', scalar_value=self.environment.scores, global_step=n)
        self.logger.add_scalar(tag='Steps', scalar_value=self.environment.steps, global_step=n)
        self.logger.add_scalar(tag='GameReward', scalar_value=np.sum(a=rewards_total), global_step=n)
        self.logger.add_images(tag='GameImages', img_tensor=game_images, dataformats='NHWC', global_step=n)
        self.logger.add_scalar(tag='NMP1', scalar_value=sizes[0], global_step=n)
        self.logger.add_scalar(tag='NMP2', scalar_value=sizes[1], global_step=n)
        self.logger.add_scalar(tag='PMP2', scalar_value=sizes[2], global_step=n)
        self.logger.add_scalar(tag='PMP1', scalar_value=sizes[3], global_step=n)
        self.logger.add_scalar(tag='SMLossLeft', scalar_value=np.sum(a=short_memory_loss_left), global_step=n)
        self.logger.add_scalar(tag='SMLossUp', scalar_value=np.sum(a=short_memory_loss_up), global_step=n)
        self.logger.add_scalar(tag='SMLossRight', scalar_value=np.sum(a=short_memory_loss_right), global_step=n)
        self.logger.add_scalar(tag='SMLossDown', scalar_value=np.sum(a=short_memory_loss_down), global_step=n)
        self.logger.add_scalar(tag='SMLossTotal', scalar_value=np.sum(a=short_memory_loss_total), global_step=n)
        self.logger.add_scalar(tag='LMLossLeft', scalar_value=lm_losses[env.LEFT], global_step=n)
        self.logger.add_scalar(tag='LMLossUp', scalar_value=lm_losses[env.UP], global_step=n)
        self.logger.add_scalar(tag='LMLossRight', scalar_value=lm_losses[env.RIGHT], global_step=n)
        self.logger.add_scalar(tag='LMLossDown', scalar_value=lm_losses[env.DOWN], global_step=n)
        self.logger.add_scalar(tag='LMLossTotal', scalar_value=np.sum(a=lm_losses), global_step=n)

        if n % 1000 == 0:
            self.agent.model.save()