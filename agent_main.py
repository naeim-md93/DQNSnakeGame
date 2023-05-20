import os
import torch
import argparse

if os.path.exists(path='env.py'):
    os.remove(path='env.py')

from lib.RL.settings import agent_settings as S
from src.utils import pyutils, debug


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Snake Game')

    # General settings
    parser.add_argument('--session', default='MMD0', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--root_path', default=os.getcwd(), type=str)
    parser.add_argument('--total_games', default=S.TOTAL_GAMES, type=int)
    parser.add_argument('--use_pygame', default=S.USE_PYGAME, action='store_true')
    parser.add_argument('--debug', default=S.DEBUG, action='store_true')

    # Environment settings
    parser.add_argument('--game_name', default='DQN Snake Game', type=str)
    parser.add_argument('--board_size', nargs='+', default=S.BOARD_SIZE, type=int)
    parser.add_argument('--cell_size', default=S.CELL_SIZE, type=int)

    # Board settings
    parser.add_argument('--background_color', nargs='+', default=S.BACKGROUND_COLOR, type=int)
    parser.add_argument('--background_value', default=S.BACKGROUND_VALUE, type=int)
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
    parser.add_argument('--min_snake_length', default=S.MIN_SNAKE_LENGTH, type=int)
    parser.add_argument('--init_snake_length', default=3, type=int)
    parser.add_argument('--snake_reduction_rate', default=0, type=int)

    # Food settings
    parser.add_argument('--food_color', nargs='+', default=S.FOOD_COLOR, type=int)
    parser.add_argument('--food_value', default=S.FOOD_VALUE, type=int)
    parser.add_argument('--min_num_foods', default=S.MIN_NUM_FOODS, type=int)
    parser.add_argument('--init_num_foods', default=1, type=int)
    parser.add_argument('--food_reduction_rate', default=0, type=int)

    # Exploration/Exploitation trade-off
    parser.add_argument('--explore_games', default=S.EXPLORE_GAMES, type=int)
    parser.add_argument('--init_epsilon', default=S.INIT_EPSILON, type=float)
    parser.add_argument('--final_epsilon', default=S.FINAL_EPSILON, type=float)

    # Important/Un-important memory trade-off
    parser.add_argument('--train_games', default=S.TRAIN_GAMES, type=int)
    parser.add_argument('--init_eta', default=S.INIT_ETA, type=float)
    parser.add_argument('--final_eta', default=S.FINAL_ETA, type=float)

    # Memory pool settings
    parser.add_argument('--max_memory', default=S.MAX_MEMORY, type=int)
    parser.add_argument('--memory_threshold', default=S.MEMORY_THRESHOLD, type=float)
    parser.add_argument('--NMP1_proportion', default=S.NMP1_PROPORTION, type=float)
    parser.add_argument('--NMP2_proportion', default=S.NMP2_PROPORTION, type=float)
    parser.add_argument('--PMP2_proportion', default=S.PMP2_PROPORTION, type=float)
    parser.add_argument('--PMP1_proportion', default=S.PMP1_PROPORTION, type=float)

    # Train settings
    parser.add_argument('--lr', default=S.LEARNING_RATE, type=float)
    parser.add_argument('--short_memory_gamma', default=S.SHORT_MEMORY_GAMMA, type=float)
    parser.add_argument('--long_memory_gamma', default=S.LONG_MEMORY_GAMMA, type=float)
    parser.add_argument('--batch_size', default=S.BATCH_SIZE, type=int)

    args = parser.parse_args()

    # Set random state
    pyutils.set_seed(seed=args.seed)

    logs_path = os.path.join(args.root_path, 'logs', args.session)
    checkpoints_path = os.path.join(args.root_path, 'checkpoints', args.session)
    os.makedirs(name=logs_path, exist_ok=True)
    os.makedirs(name=checkpoints_path, exist_ok=True)

    # Check inputs ################################################################################
    # General settings
    debug.check_int_value(x=args.total_games, minimum=1, name='total_games')

    # Environment settings
    debug.check_list_of_ints(x=args.board_size, num_elements=2, mins=[4, 4], name='board_size')
    debug.check_int_value(x=args.cell_size, minimum=1, name='cell_size')
    args.cell_size, args.display_size = debug.check_display_size(board_size=args.board_size, cell_size=args.cell_size)

    # Board settings
    debug.check_list_of_ints(x=args.background_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255],
                             name='background_color')
    debug.check_int_value(x=args.background_value, minimum=0, maximum=4, name='background_value')
    debug.check_int_value(x=args.left_index, minimum=0, maximum=3, name='left_index')
    debug.check_int_value(x=args.up_index, minimum=0, maximum=3, name='up_index')
    debug.check_int_value(x=args.right_index, minimum=0, maximum=3, name='right_index')
    debug.check_int_value(x=args.down_index, minimum=0, maximum=3, name='down_index')

    # Border settings
    debug.check_list_of_ints(x=args.border_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255],
                             name='border_color')
    debug.check_int_value(x=args.border_value, minimum=0, maximum=4, name='border_value')

    # Snake settings
    debug.check_list_of_ints(x=args.snake_head_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255],
                             name='snake_head_color')
    debug.check_int_value(x=args.snake_head_value, minimum=0, maximum=4, name='snake_head_value')
    debug.check_list_of_ints(x=args.snake_body_color, num_elements=3, mins=[0, 0, 0], maxs=[255, 255, 255],
                             name='snake_body_color')
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

    # Exploration/Exploitation trade-off
    debug.check_int_value(x=args.explore_games, minimum=1, maximum=args.train_games, name='explore_games')
    debug.check_float_value(x=args.init_epsilon, minimum=args.final_epsilon, maximum=1, name='init_epsilon')
    debug.check_float_value(x=args.final_epsilon, minimum=0, maximum=args.init_epsilon, name='final_epsilon')

    # Important/Un-important memory trade-off
    debug.check_int_value(x=args.train_games, minimum=args.explore_games, maximum=args.total_games, name='train_games')
    debug.check_float_value(x=args.init_eta, minimum=args.final_eta, maximum=1, name='init_eta')
    debug.check_float_value(x=args.final_eta, minimum=0, maximum=args.init_eta, name='final_eta')

    # Memory pool settings
    debug.check_int_value(x=args.max_memory, minimum=1000, name='max_memory')
    debug.check_float_value(x=args.memory_threshold, minimum=0, name='memory_threshold')
    debug.check_float_value(x=args.NMP1_proportion, minimum=0, maximum=1, name='NMP1_proportion')
    debug.check_float_value(x=args.NMP2_proportion, minimum=0, maximum=1, name='NMP2_proportion')
    debug.check_float_value(x=args.PMP2_proportion, minimum=0, maximum=1, name='PMP2_proportion')
    debug.check_float_value(x=args.PMP1_proportion, minimum=0, maximum=1, name='PMP1_proportion')
    assert (args.NMP1_proportion + args.NMP2_proportion + args.PMP2_proportion + args.PMP1_proportion) == 1

    # Train settings
    debug.check_float_value(x=args.lr, minimum=0, maximum=1, name='lr')
    debug.check_float_value(x=args.short_memory_gamma, minimum=0, maximum=1, name='short_memory_gamma')
    debug.check_float_value(x=args.long_memory_gamma, minimum=0, maximum=1, name='long_memory_gamma')
    debug.check_int_value(x=args.batch_size, minimum=1, name='batch_size')

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

    assert len({
        args.background_value,
        args.border_value,
        args.snake_head_value,
        args.snake_body_value,
        args.food_value
    }) == 5, f'object indices should be unique, but got ' \
             f'{args.background_value}, ' \
             f'{args.border_value}, ' \
             f'{args.snake_head_value}, ' \
             f'{args.snake_body_value}, ' \
             f'{args.food_value}'
    ###############################################################################################

    tmp = {
        # General settings
        'TOTAL_GAMES': args.total_games,
        'USE_PYGAME': args.use_pygame,
        'DEBUG': args.debug,

        # Environment settings
        'BOARD_SIZE': args.board_size,
        'CELL_SIZE': args.cell_size,
        'DISPLAY_SIZE': args.display_size,

        # Board settings
        'BACKGROUND_COLOR': args.background_color,
        'BACKGROUND_VALUE': args.background_value,
        'GRID_COLOR': S.GRID_COLOR,
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

        # Border settings
        'BORDER_COLOR': args.border_color,
        'BORDER_VALUE': args.border_value,

        # Snake settings
        'SNAKE_HEAD_COLOR': args.snake_head_color,
        'SNAKE_HEAD_VALUE': args.snake_head_value,
        'SNAKE_BODY_COLOR': args.snake_body_color,
        'SNAKE_BODY_VALUE': args.snake_body_value,
        'MIN_SNAKE_LENGTH': args.min_snake_length,
        'SNAKE_ACTIONS': {
            'LEFT': args.left_index,
            'UP': args.up_index,
            'RIGHT': args.right_index,
            'DOWN': args.down_index
        },

        # Food settings
        'FOOD_COLOR': args.food_color,
        'FOOD_VALUE': args.food_value,
        'MIN_NUM_FOODS': args.min_num_foods,

        # Exploration/Exploitation trade-off
        'EXPLORE_GAMES': args.explore_games,
        'INIT_EPSILON': args.init_epsilon,
        'FINAL_EPSILON': args.final_epsilon,

        # Important/Un-important memory trade-off
        'TRAIN_GAMES': args.train_games,
        'INIT_ETA': args.init_eta,
        'FINAL_ETA': args.final_eta,

        # Memory pool settings
        'MAX_MEMORY': args.max_memory,
        'MEMORY_THRESHOLD': args.memory_threshold,
        'NMP1_PROPORTION': args.NMP1_proportion,
        'NMP2_PROPORTION': args.NMP2_proportion,
        'PMP2_PROPORTION': args.PMP2_proportion,
        'PMP1_PROPORTION': args.PMP1_proportion,

        # Train settings
        'LEARNING_RATE': args.lr,
        'SHORT_MEMORY_GAMMA': args.short_memory_gamma,
        'LONG_MEMORY_GAMMA': args.long_memory_gamma,
        'BATCH_SIZE': args.batch_size,

        # 'REWARDS': S.REWARDS
    }

    for key, value in tmp.items():
        open(file='env.py', mode='a').write(f'{key} = {value}\n')

    from lib.RL.engines.agent_engine import AgentEngine
    from lib.RL.environment import SnakeGame
    from lib.RL.agent import QAgent

    environment = SnakeGame()
    agent = QAgent(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        checkpoints_path=checkpoints_path
    )

    engine = AgentEngine(
        environment=environment,
        agent=agent,
        init_snake_length=args.init_snake_length,
        snake_reduction_rate=args.snake_reduction_rate,
        init_num_foods=args.init_num_foods,
        food_reduction_rate=args.food_reduction_rate,
        logs_path=logs_path,
        use_pygame=args.use_pygame,
        game_name=args.game_name
    )
    engine.play_by_agent(num_games=args.total_games)
