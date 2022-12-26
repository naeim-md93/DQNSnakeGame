import pygame
import time
import argparse
import numpy as np

import CONSTANTS
from src.utils import debug
from src.RL.environment.game import SnakeGame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Snake Game')

    """ General Configs """

    """ Environment Configs """
    # Board Configs
    parser.add_argument('--board_size', nargs='+', default=[14, 14], type=int)
    parser.add_argument('--player', default='agent', type=str, choices=['user', 'agent'])

    # Pygame Configs
    parser.add_argument('--use_pygame', default=False, action='store_true')
    parser.add_argument('--pygame_game_name', default='DQN Snake Game', type=str)
    parser.add_argument('--max_obstacles', default=0, type=int)
    
    # Store inputs as module
    args = parser.parse_args()
    
    # Adding additional settings
    args.env_objects_values = CONSTANTS.ENV_OBJECTS_VALUES
    args.env_objects_colors = CONSTANTS.ENV_OBJECTS_COLORS
    args.snake_actions = CONSTANTS.SNAKE_ACTIONS
    args.board_directions = CONSTANTS.BOARD_DIRECTIONS
    
    # Check inputs
    args = debug.process_inputs(args=args)
    
    # print(args)
    env = SnakeGame(args=args)
    print(env.board.get_occupied_coords())
    print(env.board.get_empty_coords())
    print(env.board)
    time.sleep(60)
    pygame.quit()