import argparse
from typing import List, Optional


def arg_parser(default_args: Optional[List] = None):
    """Read arguments from the command line

    Args:
        default_args: None/List of arguments (seeexamples)

    Examples::
        >>> output = arg_parser()
        >>> output = arg_parser(['--model', 'PITOM',
                                '--subjects', '625', '676'])

    Miscellaneous:
        subjects: (list of strings): subject id's as a list
        shift (integer): Amount by which the onset should be shifted
        window-size (int): window size to consider for the word in ms
        bin-size (int): bin size in milliseconds
        max-electrodes (int):
        vocab-min-freq (int):

    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--electrode-list',
                       nargs='*',
                       type=int,
                       action='append')
    group.add_argument('--max-electrodes',
                       nargs='*',
                       type=int,
                       action='append')

    parser.add_argument('--subjects', nargs='+', type=str, action='append')
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--window-size', nargs='*', type=int, action='append')
    parser.add_argument('--bin-size', type=int, default=50)
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--pickle', action='store_true', default=False)

    if not default_args:
        args = parser.parse_args()
    else:
        args = parser.parse_args(default_args)

    args.subjects = [item for sublist in args.subjects for item in sublist]
    if args.electrode_list:
        args.electrode_list = [item for item in args.electrode_list if item]
        assert len(args.subjects) == len(
            args.electrode_list
        ), "Number of electrode_list calls must match number of subjects"
    else:
        args.max_electrodes = [
            item for sublist in args.max_electrodes for item in sublist
        ]
        assert len(args.subjects) == len(
            args.max_electrodes
        ), "Number of items for max-electrodes must match number of subjects"

    if not args.window_size:
        args.window_size = [2000]
    else:
        args.window_size = [
            item for sublist in args.window_size for item in sublist
        ]
        if len(args.window_size) > 2:
            raise Exception("Invalid number of window sizes")

    return args
