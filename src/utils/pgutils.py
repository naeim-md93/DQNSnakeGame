import sys

import numpy as np
import pygame


def get_user_action():
    action = np.array([0, 0, 0, 0])
    valid_action = False

    while not valid_action:
        # Quit the game if user wants to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # If a key is pressed
            if event.type == pygame.KEYDOWN:

                # If it is the left arrow key
                if event.key == pygame.K_LEFT:
                    action[0] = 1
                    valid_action = True

                elif event.key == pygame.K_UP:
                    action[1] = 1
                    valid_action = True

                elif event.key == pygame.K_DOWN:
                    action[2] = 1
                    valid_action = True

                elif event.key == pygame.K_RIGHT:
                    action[3] = 1
                    valid_action = True
    return action
