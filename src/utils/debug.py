import os
import tkinter


def check_env_inputs(args):

    # Enabling pygame if player is user
    if args.player == 'user':
        args.use_pygame = True

    color_mins = [0, 0, 0]
    color_maxs = [255, 255, 255]
    check_list_elements(x=args.pygame_cell_size, num_elements=2, parent='pygame_cell_size', mins=[1, 1])
    check_list_elements(x=args.board_size, num_elements=2, parent='board_size', mins=[4, 4])
    check_list_elements(x=args.background_color, num_elements=3, parent='background_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.grid_color, num_elements=3, parent='grid_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.border_color, num_elements=3, parent='border_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.snake_body_color, num_elements=3, parent='snake_body_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.snake_head_color, num_elements=3, parent='snake_head_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.food_color, num_elements=3, parent='food_color', mins=color_mins, maxs=color_maxs)
    check_int_value(x=args.snake_init_length, parent='snake_init_length', minimum=1, maximum=20)  # TODO: Increase Maximum
    check_int_value(x=args.num_init_foods, parent='num_init_foods', minimum=1, maximum=100)  # TODO: Increase Maximum
    args.pygame_cell_size, args.pygame_display_size = check_display_size(board_size=args.board_size, pygame_cell_size=args.pygame_cell_size)

    return args


def check_path(x, parent='input'):
    if not os.path.exists(path=x):
        raise ValueError(f'{parent} not exists at {x}')


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


def check_display_size(board_size, pygame_cell_size):
    # Get monitor size [height, width]
    tk = tkinter.Tk()
    monitor_size = [tk.winfo_screenheight(), tk.winfo_screenwidth()]

    minimum_sizes = []
    for i in range(len(board_size)):
        temp = int(monitor_size[i] // (board_size[i] * 3.5/3))
        if temp < pygame_cell_size[i]:
            minimum_sizes.append(temp)
        else:
            minimum_sizes.append(pygame_cell_size[i])

    pygame_cell_size = [min(minimum_sizes), min(minimum_sizes)]
    pygame_display_size = [board_size[0] * pygame_cell_size[0], board_size[1] * pygame_cell_size[1]]

    return pygame_cell_size, pygame_display_size
