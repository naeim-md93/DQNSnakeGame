import os
import argparse

if os.path.exists(path='env.py'):
    os.remove(path='env.py')

from lib.RL.settings import user_settings as S
from src.utils import pyutils, debug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Snake Game')

    # General settings
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--total_games', default=1, type=int)

    # Environment settings
    parser.add_argument('--game_name', default='DQN Snake Game', type=str)
    parser.add_argument('--cell_size', default=S.CELL_SIZE, type=int)
    parser.add_argument('--board_size', nargs='+', default=S.BOARD_SIZE, type=int)

    # Board settings
    parser.add_argument('--background_color', nargs='+', default=S.BACKGROUND_COLOR, type=int)
    parser.add_argument('--background_value', default=S.BACKGROUND_VALUE, type=int)
    parser.add_argument('--grid_color', nargs='+', default=S.GRID_COLOR, type=int)

    parser.add_argument('--left_index', default=S.LEFT, type=int)
    parser.add_argument('--up_index', default=S.UP, type=int)
    parser.add_argument('--right_index', default=S.RIGHT, type=int)
    parser.add_argument('--down_index', default=S.DOWN, type=int)

    # Border settings
    parser.add_argument('--border_color', nargs='+', default=S.BORDER_COLOR, type=int)
    parser.add_argument('--border_value', default=S.BORDER_VALUE, type=int)

    # Snake settings
    parser.add_argument('--snake_head_color', nargs='+', default=S.SNAKE_HEAD_COLOR, type=int)
    parser.add_argument('--snake_head_value', default=S.SNAKE_HEAD_VALUE, type=int)
    parser.add_argument('--snake_body_color', nargs='+', default=S.SNAKE_BODY_COLOR, type=int)
    parser.add_argument('--snake_body_value', default=S.SNAKE_BODY_VALUE, type=int)
    parser.add_argument('--init_snake_length', default=3, type=int)
    parser.add_argument('--min_snake_length', default=1, type=int)
    parser.add_argument('--snake_reduction_rate', default=0, type=int)

    # Food settings
    parser.add_argument('--food_color', nargs='+', default=S.FOOD_COLOR, type=int)
    parser.add_argument('--food_value', default=S.FOOD_VALUE, type=int)
    parser.add_argument('--init_num_foods', default=1, type=int)
    parser.add_argument('--min_num_foods', default=1, type=int)
    parser.add_argument('--food_reduction_rate', default=0, type=int)
    args = parser.parse_args()

    # Set random state
    pyutils.set_seed(seed=args.seed)

    # Check inputs ################################################################################
    # Check general settings
    debug.check_int_value(x=args.total_games, minimum=1, name='total_games')

    # Check environment settings
    debug.check_list_of_ints(x=args.board_size, num_elements=2, mins=[4, 4], name='board_size')
    debug.check_int_value(x=args.cell_size, minimum=1, name='cell_size')
    args.cell_size, args.display_size = debug.check_display_size(board_size=args.board_size, cell_size=args.cell_size)

    # Board settings
    debug.check_list_of_ints(x=args.background_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='background_color')
    debug.check_int_value(x=args.background_value, minimum=0, maximum=4, name='background_value')
    debug.check_list_of_ints(x=args.grid_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='grid_color')

    debug.check_int_value(x=args.left_index, minimum=0, maximum=3, name='left_index')
    debug.check_int_value(x=args.up_index, minimum=0, maximum=3, name='up_index')
    debug.check_int_value(x=args.right_index, minimum=0, maximum=3, name='right_index')
    debug.check_int_value(x=args.down_index, minimum=0, maximum=3, name='down_index')
    assert len({
        args.left_index,
        args.up_index,
        args.right_index,
        args.down_index
    }) == 4, f'direction indices should be unique, but got ' \
             f'{args.left_index}, ' \
             f'{args.up_index}, ' \
             f'{args.right_index}, ' \
             f'{args.down_index}'

    # Border settings
    debug.check_list_of_ints(x=args.border_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='border_color')
    debug.check_int_value(x=args.border_value, minimum=0, maximum=4, name='border_value')

    # Snake settings
    debug.check_list_of_ints(x=args.snake_head_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='snake_head_color')
    debug.check_int_value(x=args.snake_head_value, minimum=0, maximum=4, name='snake_head_value')
    debug.check_list_of_ints(x=args.snake_body_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='snake_body_color')
    debug.check_int_value(x=args.snake_body_value, minimum=0, maximum=4, name='snake_body_value')
    debug.check_int_value(x=args.min_snake_length, minimum=1, name='min_snake_length')
    # TODO: increase initial snake length to maximum possible
    debug.check_int_value(x=args.init_snake_length, minimum=args.min_snake_length, maximum=4, name='init_snake_length')
    debug.check_int_value(x=args.snake_reduction_rate, minimum=0, name='snake_reduction_rate')

    # Food settings
    debug.check_list_of_ints(x=args.food_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255], name='food_color')
    debug.check_int_value(x=args.food_value, minimum=0, maximum=4, name='food_value')
    debug.check_int_value(x=args.min_num_foods, minimum=1, name='min_num_foods')
    debug.check_int_value(x=args.init_num_foods, minimum=args.min_num_foods, name='init_num_foods')
    debug.check_int_value(x=args.food_reduction_rate, minimum=0, name='food_reduction_rate')

    assert len({
        args.background_value,
        args.border_value,
        args.snake_head_value,
        args.snake_body_value,
        args.food_value
    }) == 5, f'object indices should be unique, but got '\
        f'{args.background_value}, ' \
        f'{args.border_value}, ' \
        f'{args.snake_head_value}, ' \
        f'{args.snake_body_value}, ' \
        f'{args.food_value}'
    ###############################################################################################

    # Update reinforcement library settings
    tmp = {
        'CELL_SIZE': args.cell_size,
        'BOARD_SIZE': args.board_size,
        'DISPLAY_SIZE': args.display_size,
        'DEBUG': S.DEBUG,
        'BACKGROUND_COLOR': args.background_color,
        'BACKGROUND_VALUE': args.background_value,
        'GRID_COLOR': args.grid_color,
        'LEFT': args.left_index,
        'UP': args.up_index,
        'RIGHT': args.right_index,
        'DOWN': args.down_index,
        'BOARD_DIRECTIONS': {
            args.left_index: {'ADD_TO_COORDS': (0, -1), 'INVALID': args.right_index},  # (+h, +w, index)
            args.up_index: {'ADD_TO_COORDS': (-1, 0), 'INVALID': args.down_index},  # (+h, +w, index)
            args.right_index: {'ADD_TO_COORDS': (0, 1), 'INVALID': args.left_index},   # (+h, +w, index)
            args.down_index: {'ADD_TO_COORDS': (1, 0), 'INVALID': args.up_index}    # (+h, +w, index)
        },
        'BORDER_VALUE': args.border_value,
        'BORDER_COLOR': args.border_color,
        'SNAKE_BODY_VALUE': args.snake_body_value,
        'SNAKE_BODY_COLOR': args.snake_body_color,
        'SNAKE_HEAD_VALUE': args.snake_head_value,
        'SNAKE_HEAD_COLOR': args.snake_head_color,
        'SNAKE_ACTIONS': {
            'LEFT': args.left_index,
            'UP': args.up_index,
            'RIGHT': args.right_index,
            'DOWN': args.down_index
        },
        'MIN_SNAKE_LENGTH': args.min_snake_length,
        'FOOD_VALUE': args.food_value,
        'FOOD_COLOR': args.food_color,
        'MIN_NUM_FOODS': args.min_num_foods,
    }

    for key, value in tmp.items():
        open(file='env.py', mode='a').write(f'{key} = {value}\n')

    from lib.RL.environment import SnakeGame
    from lib.RL.engines.user_engine import UserEngine

    # Initialize environment
    environment = SnakeGame()

    # Initialize engine
    engine = UserEngine(
        environment=environment,
        game_name=args.game_name,
        init_snake_length=args.init_snake_length,
        snake_reduction_rate=args.snake_reduction_rate,
        init_num_foods=args.init_num_foods,
        food_reduction_rate=args.food_reduction_rate,
    )

    # Play
    engine.play_by_user(num_games=args.total_games)
