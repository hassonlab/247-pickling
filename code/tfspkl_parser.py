import argparse
import os
from typing import List, Optional


def arg_parser(default_args: Optional[List] = None):
    """Read arguments from the command line

    Args:
        default_args: None/List of arguments (see examples)

    Examples::
        >>> output = arg_parser()
        >>> output = arg_parser(['--model', 'PITOM',
                                '--subjects', '625', '676'])

    Miscellaneous:
        subjects: (list of strings): subject id's as a list
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
    group.add_argument('--max-electrodes', type=int, default=250)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--subject', type=int, default=0)
    group1.add_argument('--sig-elec-file', type=str, default='')

    parser.add_argument('--bin-size', type=int, default=32)
    parser.add_argument('--vocab-min-freq', type=int, default=0)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--project-id', type=str, default=None)

    if not default_args:
        args = parser.parse_args()
    else:
        args = parser.parse_args(default_args)

    if not args.subject:
        args.subject = 777

    return args
