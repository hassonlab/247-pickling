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

    parser.add_argument('--subject', type=int, default=0)
    parser.add_argument('--bin-size', type=int, default=32)
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--pickle', action='store_true', default=False)
    parser.add_argument('--num-folds', type=int, default=5)

    if not default_args:
        args = parser.parse_args()
    else:
        args = parser.parse_args(default_args)

    if args.electrode_list:
        args.electrode_list = [item for item in args.electrode_list if item]
    else:
        args.max_electrodes = [
            item for sublist in args.max_electrodes for item in sublist
        ]

    return args
