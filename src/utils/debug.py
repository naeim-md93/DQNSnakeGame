import tkinter


def check_int_value(x: int, minimum: int = None, maximum: int = None, name: str = 'input') -> None:
    """
    Check int value
    :param x: value
    :param minimum: minimum value
    :param maximum: maximum value
    :param name: name of the variable
    :return: None
    """
    if not isinstance(x, int):
        raise ValueError(f'{name} should be of type int, but got {type(x)}')

    if (minimum is not None) and (x < minimum):
        raise ValueError(f'{name} value should be at least {minimum}, but got {x}')

    if (maximum is not None) and (x > maximum):
        raise ValueError(f'{name} value should be at most {maximum}, but got {x}')


def check_float_value(x: int, minimum: int = None, maximum: int = None, name: str = 'input') -> None:
    """
    Check int value
    :param x: value
    :param minimum: minimum value
    :param maximum: maximum value
    :param name: name of the variable
    :return: None
    """
    if not isinstance(x, float):
        raise ValueError(f'{name} should be of type int, but got {type(x)}')

    if (minimum is not None) and (x < minimum):
        raise ValueError(f'{name} value should be at least {minimum}, but got {x}')

    if (maximum is not None) and (x > maximum):
        raise ValueError(f'{name} value should be at most {maximum}, but got {x}')


def check_list_of_ints(
        x: list[int],
        num_elements: int,
        name: str = 'input',
        mins: list[int] = None,
        maxs: list[int] = None
) -> None:
    """
    Checking a list of elements
    First check if number of elements in list is equal to num_elements
    then check each element using check_int_value function
    :param x: list of ints
    :param num_elements: number of elements in x
    :param name: name of the variable containing x
    :param mins: minimum value for each element in x
    :param maxs: maximum value for each element in x
    :return:
    """
    if not isinstance(x, list):
        raise ValueError(f'{name} should be of type list, but got {type(x)}')

    if len(x) != num_elements:
        raise ValueError(f'{name} should have {num_elements} elements, but got {len(x)}')

    for i, c in enumerate(x):
        if mins is not None:
            check_int_value(x=c, minimum=mins[i], maximum=None, name=name)

        if maxs is not None:
            check_int_value(x=c, minimum=None, maximum=maxs[i], name=name)


def check_display_size(board_size: list[int], cell_size: int) -> tuple[int, list[int, int]]:
    """
    Check if board_size with cell_size can be displayed on user monitor
    :param board_size: Size of the board game
    :param cell_size: size of each cell in board game
    :return: adjusted cell_size and display size
    """

    # Get monitor size [height, width]
    tk = tkinter.Tk()
    # Get minimum display size;
    # -200 for taskbar (in bottom) and notification bar (in top)
    min_disp_size = min([tk.winfo_screenheight(), tk.winfo_screenwidth()]) - 200

    # For each height and width in board size
    for i in range(len(board_size)):

        # Reduce cell size until board size with cell size fits in display size
        while ((board_size[i] * cell_size) + board_size[i] + 1) > min_disp_size:
            cell_size = cell_size - 1

    # (board_size[i] + 1 for lines between each cell
    display_size = [
        board_size[0] * cell_size + board_size[0] + 1,
        board_size[1] * cell_size + board_size[1] + 1
    ]

    if min(display_size) < 1:
        raise ValueError(f"Can't start pygame with board_size {board_size} and cell size {cell_size}")

    return cell_size, display_size
