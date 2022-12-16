import argparse
from tqdm import trange
from src.RL.environments.games.snake_game import SnakeGame
from src.utils import pgutils, debug


WHITE = [255, 255, 255]  # Background
DARK_GREEN = [0, 155, 0]  # Snake Body
GREEN = [0, 255, 0]  # Snake Head
RED = [255, 0, 0]  # Apple
BLACK = [0, 0, 0]  # Borders
DARK_GRAY = (100, 100, 100) # Grid


class Engine:
    def __init__(self, args, env):
        self.args = args
        self.env = env

    def user_play_one_game(self):
        game_over = False
        self.env.reset(num_foods=self.args.num_init_foods, snake_length=self.args.snake_init_length)
        self.env.display()
        while not game_over:
            game_over = self.user_play_one_step()
            self.env.display()
            print(f'Score: {self.env.scores}, Game Over: {game_over}, Empty: {self.env.board.get_empty_blocks()}')

    def user_play_one_step(self):
        action = pgutils.get_user_action()
        game_over = self.env.move(action)
        return game_over

    def play(self):
        for t in trange(self.args.total_games):
            self.user_play_one_game()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Snake Game')

    """ General Configs """
    parser.add_argument('--player', default='agent', type=str, choices=['user', 'agent'])
    parser.add_argument('--total_games', default=10, type=int)

    """ Environment Configs """
    # Pygame Configs
    parser.add_argument('--use_pygame', default=False, action='store_true')
    parser.add_argument('--pygame_game_name', default='DQN Snake Game', type=str)
    parser.add_argument('--pygame_cell_size', nargs='+', default=[20, 20], type=int)

    # Board Configs
    parser.add_argument('--board_size', nargs='+', default=[14, 14], type=int)
    parser.add_argument('--background_value', default=0, type=int)
    parser.add_argument('--background_color', nargs='+', default=WHITE, type=int)
    parser.add_argument('--grid_color', nargs='+', default=DARK_GRAY, type=int)

    # Border Configs
    parser.add_argument('--border_value', default=1, type=int)
    parser.add_argument('--border_color', nargs='+', default=BLACK, type=int)

    # Snake Configs
    parser.add_argument('--snake_body_value', default=2, type=int)
    parser.add_argument('--snake_body_color', nargs='+', default=GREEN, type=int)
    parser.add_argument('--snake_head_value', default=3, type=int)
    parser.add_argument('--snake_head_color', nargs='+', default=DARK_GREEN, type=int)
    parser.add_argument('--snake_init_length', default=3, type=int)

    # Food Configs
    parser.add_argument('--food_value', default=4, type=int)
    parser.add_argument('--food_color', nargs='+', default=RED, type=int)
    parser.add_argument('--num_init_foods', default=1, type=int)

    args = parser.parse_args()

    # Check environment inputs
    args = debug.check_env_inputs(args=args)

    print(args)

    env = SnakeGame(args=args)
    engine = Engine(args=args, env=env)
    engine.play()