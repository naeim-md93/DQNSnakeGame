import CONSTANTS


def process_inputs(args):
    
    # Use pygame if player is user
    if args.player == 'user':
        args.use_pygame = True
    
    color_mins = [0, 0, 0]
    color_maxs = [255, 255, 255]
    
    # TODO: Board size less than (4, 4) (whitout border)
    check_list_elements(x=args.board_size, num_elements=2, parent='board_size', mins=[4, 4])

    check_list_elements(x=args.background_color, num_elements=3, parent='background_color', mins=color_mins, maxs=color_maxs)
    check_list_elements(x=args.grid_color, num_elements=3, parent='grid_color', mins=color_mins, maxs=color_maxs)
    
    # Processing display settings
    args = process_displaying_settings(args=args)
    
    return args


def check_int_value(x, minimum=None, maximum=None, parent='input'):
    if x is not None:
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


def process_displaying_settings(args):
    max_pygame_board_size = [
        (CONSTANTS.MAX_DISP_SIZE[0] - 1) // 2,
        (CONSTANTS.MAX_DISP_SIZE[1] - 1) // 2,
    ]
    print(max_pygame_board_size)
    
    if args.board_size[0] > max_pygame_board_size[0] or args.board_size[1] > max_pygame_board_size[1]:    
        if args.player == 'user':
            raise ValueError(f'For using pygame, board size should be at most {max_pygame_board_size}')
        else:
            args.use_pygame = False
        
    
    tmp_cell_size = [
        ((CONSTANTS.MAX_DISP_SIZE[0] - 1) // args.board_size[0]) - 1,
        ((CONSTANTS.MAX_DISP_SIZE[1] - 1) // args.board_size[1]) - 1,
    ]
    print(tmp_cell_size)

    args.pygame_cell_size = [min(tmp_cell_size), min(tmp_cell_size)]
        
    args.pygame_display_size = [
        (args.board_size[0] * (args.pygame_cell_size[0] + 1)) + 1,
        (args.board_size[1] * (args.pygame_cell_size[1] + 1)) + 1
    ]
    
    
    return args
