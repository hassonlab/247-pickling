import argparse
from typing import List, Optional


def arg_parser(default_args: Optional[List] = None):
    '''Read arguments from the command line

    Args:
        default_args: None/List of arguments (seeexamples)

    Examples::
        >>> output = arg_parser()
        >>> output = arg_parser(['--model', 'PITOM',
                                '--subjects', '625', '676'])

    Miscellaneous:
        model: DNN model to choose from (PITOM, ConvNet, MeNTALmini, MeNTAL)
        subjects: (list of strings): subject id's as a list
        shift (integer): Amount by which the onset should be shifted
        lr (float): learning rate
        gpus (int): number of gpus for the model to run on
        epochs (int): number of epochs
        batch-size (int): bach-size
        window-size (int): window size to consider for the word in ms
        bin-size (int): bin size in milliseconds
        init-model:
        no-plot (bool):
        max-electrodes (int):
        vocab-min-freq (int):
        vocab-max-freq (int):
        max-num-bins (int): upper threshold for signal length (counted as bins)
        seed (int): random seed for reproducibility
        shuffle (bool): shuffle samples (augmentation?)
        no-eval (bool): train mode only or not
        temp (float): temperature
        tf-dmodel (int): transformer hidden units
        tf-dff (int): tranfsformer feed forward units
        tf-dhead (int): transformer number of heads
        tf-nlayer (int): transformer encoder/decoder layers
        tf-dropout (float): dropout rate
        weight-decay (int):

        #TODO: maybe remove some input arguments
'''
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

    parser.add_argument('--model', type=str, default='MeNTAL')
    parser.add_argument('--subjects', nargs='+', type=str, action='append')
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=48)
    # parser.add_argument('--window-size', type=int, default=2000)
    parser.add_argument('--window-size', nargs='*', type=int, action='append')
    parser.add_argument('--bin-size', type=int, default=50)
    parser.add_argument('--init-model', type=str, default=None)
    parser.add_argument('--no-plot', action='store_false', default=False)
    parser.add_argument('--ngrams', action='store_true', default=False)
    parser.add_argument('--nseq', action='store_true', default=False)
    parser.add_argument('--seq-len-limit', type=int, default=75)
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--vocab-max-freq', type=int, default=1000000)
    parser.add_argument('--max-num-bins', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--no-eval', action="store_true", default=False)
    parser.add_argument('--temp', type=float, default=0.995)
    parser.add_argument('--tf-dmodel', type=int, default=64)
    parser.add_argument('--tf-dff', type=int, default=128)
    parser.add_argument('--tf-nhead', type=int, default=4)
    parser.add_argument('--tf-nlayer', type=int, default=3)
    parser.add_argument('--tf-dropout', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.35)
    parser.add_argument('--output-folder', type=str, default=None)
    parser.add_argument('--exp-suffix', type=str, default='NAE')
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
