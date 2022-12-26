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

    
    parser.add_argument('--background_color', nargs='+', default=CONSTANTS.BACKGROUND_COLOR, type=int)
    parser.add_argument('--grid_color', nargs='+', default=CONSTANTS.GRID_COLOR, type=int)
    
    # Store inputs as module
    args = parser.parse_args()
    
    # Check inputs
    args = debug.process_inputs(args=args)
    
    print(args)
    env = SnakeGame(args=args)
    
    time.sleep(10)
    pygame.quit()