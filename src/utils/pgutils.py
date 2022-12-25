import sys
import pygame
import numpy as np

import CONSTANTS


def get_user_action(snake_direction):
    action = np.zeros(shape=(len(CONSTANTS.BOARD_DIRECTIONS), ))
    valid_action = False

    while not valid_action:
        # Quit the game if user wants to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # If a key is pressed
            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_LEFT:
                    if snake_direction != CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.LEFT]['INVALID']:
                        action[CONSTANTS.LEFT] = 1
                        valid_action = True

                elif event.key == pygame.K_UP:
                    if snake_direction != CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.UP]['INVALID']:
                        action[CONSTANTS.UP] = 1
                        valid_action = True

                elif event.key == pygame.K_RIGHT:
                    if snake_direction != CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.RIGHT]['INVALID']:
                        action[CONSTANTS.RIGHT] = 1
                        valid_action = True

                elif event.key == pygame.K_DOWN:
                    if snake_direction != CONSTANTS.BOARD_DIRECTIONS[CONSTANTS.DOWN]['INVALID']:
                        action[CONSTANTS.DOWN] = 1
                        valid_action = True

    return action


def get_next_head_coords(snake_actions, board_directions):

    left = (0, -1, board_directions['LEFT'])
    up = (-1, 0, board_directions['UP'])
    right = (0, 1, board_directions['RIGHT'])
    down = (1, 0, board_directions['DOWN'])

    if len(snake_actions) == 3:
        nhc = ([], [], [])

        nhc[snake_actions['TURN_LEFT']].insert(board_directions['LEFT'], down)
        nhc[snake_actions['TURN_LEFT']].insert(board_directions['UP'], left)
        nhc[snake_actions['TURN_LEFT']].insert(board_directions['RIGHT'], up)
        nhc[snake_actions['TURN_LEFT']].insert(board_directions['DOWN'], right)

        nhc[snake_actions['GO_STRAIGHT']].insert(board_directions['LEFT'], left)
        nhc[snake_actions['GO_STRAIGHT']].insert(board_directions['UP'], up)
        nhc[snake_actions['GO_STRAIGHT']].insert(board_directions['RIGHT'], right)
        nhc[snake_actions['GO_STRAIGHT']].insert(board_directions['DOWN'], down)

        nhc[snake_actions['TURN_RIGHT']].insert(board_directions['LEFT'], up)
        nhc[snake_actions['TURN_RIGHT']].insert(board_directions['UP'], right)
        nhc[snake_actions['TURN_RIGHT']].insert(board_directions['RIGHT'], down)
        nhc[snake_actions['TURN_RIGHT']].insert(board_directions['DOWN'], left)

    elif len(snake_actions) == 4:

        nhc = ([], [], [], [])

        nhc[snake_actions['LEFT']].insert(board_directions['LEFT'], left)
        nhc[snake_actions['LEFT']].insert(board_directions['UP'], left)
        nhc[snake_actions['LEFT']].insert(board_directions['DOWN'], left)
        nhc[snake_actions['LEFT']].insert(board_directions['RIGHT'], None)

        nhc[snake_actions['UP']].insert(board_directions['LEFT'], up)
        nhc[snake_actions['UP']].insert(board_directions['UP'], up)
        nhc[snake_actions['UP']].insert(board_directions['DOWN'], None)
        nhc[snake_actions['UP']].insert(board_directions['RIGHT'], up)

        nhc[snake_actions['RIGHT']].insert(board_directions['LEFT'], None)
        nhc[snake_actions['RIGHT']].insert(board_directions['UP'], right)
        nhc[snake_actions['RIGHT']].insert(board_directions['DOWN'], right)
        nhc[snake_actions['RIGHT']].insert(board_directions['RIGHT'], right)

        nhc[snake_actions['DOWN']].insert(board_directions['LEFT'], down)
        nhc[snake_actions['DOWN']].insert(board_directions['UP'], None)
        nhc[snake_actions['DOWN']].insert(board_directions['DOWN'], down)
        nhc[snake_actions['DOWN']].insert(board_directions['RIGHT'], down)

    else:
        raise ValueError(f'Invalid snake actions')

    return nhc


def get_next_snake_action(board_directories, snake_actions):

    nsa = [[], [], [], []]

    nsa[board_directories['LEFT']].insert(board_directories['LEFT'], 1)
    nsa[board_directories['LEFT']].insert(board_directories['UP'], 2)
    nsa[board_directories['LEFT']].insert(board_directories['RIGHT'], None)
    nsa[board_directories['LEFT']].insert(board_directories['DOWN'], 0)

    nsa[board_directories['UP']].insert(board_directories['LEFT'], 0)
    nsa[board_directories['UP']].insert(board_directories['UP'], 1)
    nsa[board_directories['UP']].insert(board_directories['RIGHT'], 2)
    nsa[board_directories['UP']].insert(board_directories['DOWN'], None)

    nsa[board_directories['RIGHT']].insert(board_directories['LEFT'], None)
    nsa[board_directories['RIGHT']].insert(board_directories['UP'], 0)
    nsa[board_directories['RIGHT']].insert(board_directories['RIGHT'], 1)
    nsa[board_directories['RIGHT']].insert(board_directories['DOWN'], 2)

    nsa[board_directories['DOWN']].insert(board_directories['LEFT'], 2)
    nsa[board_directories['DOWN']].insert(board_directories['UP'], None)
    nsa[board_directories['DOWN']].insert(board_directories['RIGHT'], 0)
    nsa[board_directories['DOWN']].insert(board_directories['DOWN'], 1)

    return nsa
