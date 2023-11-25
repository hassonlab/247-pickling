import argparse
import sys


def arg_parser():
    """Read arguments from the command line

    subjects: (list of strings): subject id's as a list
    bin-size (int): bin size in milliseconds
    max-electrodes (int): Specify a large number so all electrodes are read
    vocab-min-freq (int): minimum number of words to keep

    """
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--electrode-list", nargs="*", type=int, action="append")
    group.add_argument("--max-electrodes", type=int, default=1e4)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--subject", type=str, default=None)
    group1.add_argument("--sig-elec-file", type=str, default="")

    parser.add_argument("--bin-size", type=int, default=32)
    parser.add_argument("--vocab-min-freq", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--project-id", type=str, default="podcast")

    # If running the code in debug mode
    gettrace = getattr(sys, "gettrace", None)

    if gettrace():
        sys.argv = [
            "scripts/tfspkl_parser.py",
            "--project-id",
            "tfs",
            "--subject",
            "625",
            "--max-electrodes",
            "500",
        ]

    args = parser.parse_args()

    if not args.subject:
        args.subject = "777"

    return args
