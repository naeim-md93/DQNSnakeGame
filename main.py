import os
import torch
import argparse
from tqdm import trange

import CONSTANTS
from src.utils import pgutils, debug
from src.RL.environments.games.snake_game import SnakeGame
from src.RL.agents.agent import Agent


class Engine:
    def __init__(self, args, env, agent):
        self.env = env
        self.agent = agent
        self.total_games = args.total_games

        self.explore_games = args.explore_games
        self.init_epsilon = args.init_epsilon
        self.final_epsilon = args.final_epsilon

        self.player_is_agent = True if args.player == 'agent' else False
        self.num_init_foods = args.num_init_foods
        self.snake_init_length = args.snake_init_length

    def play_one_step(self, **kwargs):
        if self.player_is_agent:
            action = self.agent.get_agent_action(**kwargs)
        else:
            action = pgutils.get_user_action(snake_direction=self.env.board.snake.direction)

        game_over = self.env.move_one_step(action)

        return game_over

    def play_one_game(self, g, **kwargs):
        game_over = False

        self.env.reset(num_foods=self.num_init_foods, snake_length=self.snake_init_length)

        if self.player_is_agent:
            state0, legal_moves0, img = self.env.get_step_info()
            kwargs['legal_moves0'] = legal_moves0
            kwargs['state0'] = state0
        else:
            self.env.display()

        while not game_over:
            game_over = self.play_one_step(**kwargs)

    def play(self):
        kwargs = {}
        if self.player_is_agent:
            kwargs['epsilon'] = self.init_epsilon

        for g in trange(1, self.total_games + 1):
            self.play_one_game(g=g, **kwargs)

            if self.player_is_agent:
                if g > self.explore_games:
                    kwargs['epsilon'] = self.final_epsilon
                else:
                    tmp = (self.final_epsilon - self.init_epsilon) / self.explore_games
                    kwargs['epsilon'] = kwargs['epsilon'] + tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Snake Game')

    """ General Configs """
    parser.add_argument('--session', default='Try1', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--root_path', default=os.getcwd(), type=str)
    parser.add_argument('--player', default='agent', type=str, choices=['user', 'agent'])
    parser.add_argument('--total_games', default=100, type=int)

    """ Environment Configs """
    # Pygame Configs
    parser.add_argument('--use_pygame', default=False, action='store_true')
    parser.add_argument('--pygame_game_name', default='DQN Snake Game', type=str)
    parser.add_argument('--pygame_cell_size', nargs='+', default=[20, 20], type=int)

    # Board Configs
    parser.add_argument('--board_size', nargs='+', default=[14, 14], type=int)
    parser.add_argument('--background_value', default=0, type=int)
    parser.add_argument('--background_color', nargs='+', default=CONSTANTS.WHITE, type=int)
    parser.add_argument('--grid_color', nargs='+', default=CONSTANTS.DARK_GRAY, type=int)

    # Border Configs
    parser.add_argument('--border_value', default=1, type=int)
    parser.add_argument('--border_color', nargs='+', default=CONSTANTS.BLACK, type=int)

    # Snake Configs
    parser.add_argument('--num_snake_actions', default=4, type=int, choices=[4])
    parser.add_argument('--snake_init_length', default=3, type=int)
    parser.add_argument('--snake_body_value', default=2, type=int)
    parser.add_argument('--snake_body_color', nargs='+', default=CONSTANTS.GREEN, type=int)
    parser.add_argument('--snake_head_value', default=3, type=int)
    parser.add_argument('--snake_head_color', nargs='+', default=CONSTANTS.DARK_GREEN, type=int)

    # Food Configs
    parser.add_argument('--num_init_foods', default=1, type=int)
    parser.add_argument('--food_value', default=4, type=int)
    parser.add_argument('--food_color', nargs='+', default=CONSTANTS.RED, type=int)

    """ Agent """
    parser.add_argument('--explore_games', default=10, type=int)
    parser.add_argument('--init_epsilon', default=0.5, type=int)
    parser.add_argument('--final_epsilon', default=0.0, type=int)


    args = parser.parse_args()

    # Check inputs
    args = debug.check_inputs(args=args)

    # Additional snake settings
    args.snake_actions = CONSTANTS.SNAKE_ACTIONS

    # Additional board settings
    args.board_directions = CONSTANTS.BOARD_DIRECTIONS

    # Enabling pygame if player is user
    if args.player == 'user':
        args.use_pygame = True

    # Logs path and Models path
    args.logs_path = os.path.join(args.root_path, 'checkpoints', args.session, 'logs')
    args.models_path = os.path.join(args.root_path, 'checkpoints', args.session, 'models')

    os.makedirs(name=args.logs_path, exist_ok=True)
    os.makedirs(name=args.models_path, exist_ok=True)

    # device
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = torch.device('cpu')
    print(args)

    env = SnakeGame(args=args)
    agent = Agent(args=args)
    engine = Engine(args=args, env=env, agent=agent)
    engine.play()
