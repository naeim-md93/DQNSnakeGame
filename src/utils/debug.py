import os

def check_env_inputs(args):
    color_mins = [0, 0, 0]
    color_maxs = [255, 255, 255]
    check_list_elements(x=args.pygame_cell_size, num_elements=2, parent='pygame_cell_size')
    check_list_elements(x=args.board_size, num_elements=2, parent='board_size')
    check_list_elements(x=args.background_color, num_elements=3, parent='background_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.grid_color, num_elements=3, parent='grid_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.border_color, num_elements=3, parent='border_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.snake_body_color, num_elements=3, parent='snake_body_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.snake_head_color, num_elements=3, parent='snake_head_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.food_color, num_elements=3, parent='food_color', mins=color_mins, maxs=color_maxs)
    check_int_value(x=args.snake_init_length, minimum=1, parent='snake_init_length')
    check_int_value(x=args.num_init_foods, minimum=1, parent='num_init_foods')

# def check_dtypes(x, dtype, parent='input', minimum=None, maximum=None):
#     if not isinstance(x, dtype):
#         raise ValueError(f'{parent} should be of type {type(dtype)}, but got {type(x)}')
#
#     if (minimum is not None) and (x < minimum):
#         raise ValueError(f'{parent} value should be at least {minimum}, but got {x}')
#
#     if (maximum is not None) and (x > maximum):
#         raise ValueError(f'{parent} value should be at most {maximum}, but got {x}')
#
#
# def check_path(x, parent='input'):
#     if not os.path.exists(path=x):
#         raise ValueError(f'{parent} not exists at {x}')


def check_int_value(x, minimum=None, maximum=None, parent='input'):
    if (minimum is not None) and (x < minimum):
        raise ValueError(f'{parent} value should be at least {minimum}, but got {x}')

    if (maximum is not None) and (x > maximum):
        raise ValueError(f'{parent} value should be at most {maximum}, but got {x}')


def check_list_elements(x, num_elements, parent='input', mins=None, maxs=None):
    if len(x) != num_elements:
        raise ValueError(f'{parent} should have {num_elements} elements, but got {len(x)}')

    for i, c in enumerate(x):
        if mins is not None:
            check_int_value(x=c, minimum=mins[i], maximum=None, parent=parent)

        if maxs is not None:
            check_int_value(x=c, minimum=None, maximum=maxs[i], parent=parent)
