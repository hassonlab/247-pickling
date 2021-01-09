import glob
import math
import os
import random

import mat73
import numpy as np
import pandas as pd
import torch


def read_file(fh):
    """Read file line-by-line and store in list

    Args:
        fh (string): name of the file

    Returns:
        list: contents of the file
    """
    with open(fh, 'r') as f:
        lines = [line.rstrip() for line in f]
    print(f'Number of Conversations is: {len(lines)}')
    return lines


def fix_random_seed(CONFIG):
    """Fix random seed

    Args:
        CONFIG (dict): configuration information
    """
    SEED = CONFIG['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def extract_elec_ids(conversation):
    """[summary]

    Args:
        conversation ([type]): [description]

    Returns:
        [type]: [description]
    """
    elec_files = glob.glob(os.path.join(conversation, 'preprocessed', '*.mat'))
    elec_files = sorted(
        elec_files, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))

    elec_ids_list = list(
        map(lambda x: int(os.path.splitext(x)[0].split('_')[-1]), elec_files))

    return elec_ids_list


def update_convs(convs):
    """[summary]

    Args:
        convs ([type]): [description]

    Returns:
        [type]: [description]
    """
    all_elec_ids_list = []
    all_elec_labels_list = []
    for conversation, *_, electrodes in convs:

        elect_ids_list = extract_elec_ids(conversation)
        elect_labels_list = load_header(conversation)

        if not electrodes or len(electrodes) > len(elect_ids_list):
            electrodes = elect_ids_list

        all_elec_ids_list.append(electrodes)
        all_elec_labels_list.append(elect_labels_list)

    common_electrodes = list(set.intersection(*map(set, all_elec_ids_list)))
    # raise Exception(all_elec_labels_list)
    common_labels = sorted(list(
        set.intersection(*map(set, all_elec_labels_list))),
                           key=lambda x: all_elec_labels_list[0].index(x))

    common_labels = [common_labels[elec - 1] for elec in common_electrodes]

    convs = [(*conv[:3], common_electrodes, common_labels) for conv in convs]
    return convs


def return_conversations(CONFIG):
    """Returns list of conversations

    Arguments:
        CONFIG {dict} -- Configuration information
        set_str {string} -- string indicating set type (train or valid)

    Returns:
        list -- List of tuples (directory, file, idx)
    """
    conversations = []

    for conv_dir in CONFIG["CONV_DIRS"]:
        conversation_list = [
            os.path.basename(x) for x in sorted(
                glob.glob(os.path.join(conv_dir, '*conversation*')))
        ]
        conversations.append(conversation_list)

    convs = [
        (conv_dir + conv_name, '/misc/*datum_%s.txt' % ds, idx, electrode_list)
        for idx, (conv_dir, convs, ds, electrode_list) in enumerate(
            zip(CONFIG["CONV_DIRS"], conversations, CONFIG["datum_suffix"],
                CONFIG["electrode_list"])) for conv_name in convs
    ]
    convs = update_convs(convs)

    return convs


def return_examples(file, delim, ex_words, vocab_str='std'):
    """Parse conversations to return examples

    Args:
        file (str): conversation file
        delim (str): text delimiter in the conversation file
        ex_words (list): words to be excluded
        vocab_str (str, optional): vocabulary to use. (Standard/SentencePiece)
                                   defaults to 'std'.

    Returns:
        list: example tuples
    """
    with open(file, 'r') as fin:
        lines = map(lambda x: x.split(delim), fin)
        examples = map(
            lambda x: (" ".join([
                z for y in x[0:-4]
                if (z := y.lower().strip().replace('"', '')) not in ex_words
            ]), x[-1].strip(), x[-4], x[-3], x[-2]), lines)
        examples = filter(lambda x: len(x[0]) > 0, examples)
        if vocab_str == 'spm':
            examples = map(
                lambda x:
                (vocabulary.EncodeAsIds(x[0]), x[1], int(float(x[2])),
                 int(float(x[3])), int(float(x[4]))), examples)
        elif vocab_str == 'std':
            examples = map(
                lambda x: (x[0].split(), x[1], int(float(x[2])),
                           int(float(x[3])), int(float(x[4]))), examples)
            examples = filter(lambda x: len(x[0]) == 1, examples)
        else:
            print("Bad vocabulary string")
        return list(examples)


def return_examples_new(file, delim, ex_words, vocab_str='std'):
    df = pd.read_csv(file,
                     sep=' ',
                     header=None,
                     names=['word', 'onset', 'offset', 'accuracy', 'speaker'])
    df['word'] = df['word'].str.split()
    df = df.explode('word', ignore_index=True)
    df = df[~df['word'].isin(ex_words)]

    return df.values.tolist()


def calculate_windows_params(CONFIG, gram, param_dict):
    """Add offset to window begin, end and count bins

    Args:
        CONFIG (dict): configuration information
        gram (tuple): tuple object of word and its corresponding signal window
        param_dict (dict): signal parameters

    Returns:
        int: sequence length
        int: index for window start
        int: index for window end
        int: number of bins in that window
    """
    seq_length = gram[3] - gram[2]
    begin_window = gram[2] + param_dict['start_offset']
    end_window = gram[3] + param_dict['end_offset']

    if CONFIG["classify"] and CONFIG["ngrams"] and not CONFIG["nseq"]:
        num_bins = 50
    elif CONFIG["classify"] and not CONFIG["ngrams"] and not CONFIG[
            "nseq"]:  # fixed number of bins
        left_window = param_dict["left_window"]
        right_window = param_dict["right_window"]
        num_bins = len(range(-left_window, right_window, param_dict["bin_fs"]))
    else:
        num_bins = int(
            math.ceil((end_window - begin_window) / param_dict['bin_fs']))

    return seq_length, begin_window, end_window, num_bins


def convert_ms_to_fs(CONFIG, fs=512):
    """Convert seconds to frames

    Args:
        CONFIG (dict): Configuration information
        fs (int, optional): Frames per second. Defaults to 512.

    Returns:
        dict: parameters for extracting neural signals
    """
    window_ms = CONFIG["window_size"]
    if len(window_ms) == 1:
        window_fs = int(window_ms[0] / 1000 * fs)
        half_window = window_fs // 2
        left_window = half_window
        right_window = half_window
    else:
        left_window = int(window_ms[0] / 1000 * fs)
        right_window = int(window_ms[1] / 1000 * fs)

    shift_ms = CONFIG["shift"]
    bin_ms = CONFIG["bin_size"]

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    start_offset = -left_window + shift_fs
    end_offset = right_window + shift_fs

    signal_param_dict = dict()
    signal_param_dict['bin_fs'] = bin_fs
    signal_param_dict['shift_fs'] = shift_fs
    # signal_param_dict['window_fs'] = window_fs
    signal_param_dict['left_window'] = left_window
    signal_param_dict['right_window'] = right_window
    signal_param_dict['start_offset'] = start_offset
    signal_param_dict['end_offset'] = end_offset

    return signal_param_dict


def test_for_bad_window(start, stop, shape):
    # if the window_begin is less than 0 or
    # check if onset is within limits
    # if the window_end is less than 0 or
    # if the window_end is outside the signal
    # if there are not enough frames in the window
    return (start < 0 or start > shape[0] or stop < 0 or stop > shape[0]
            or stop - start < 0)


def print_cuda_usage(CONFIG):
    """CUDA usage statistics

    Args:
        CONFIG (dict): configuration information
    """
    print('Memory Usage:')
    for i in range(CONFIG["gpus"]):
        max_alloc = round(torch.cuda.max_memory_allocated(i) / 1024**3, 1)
        cached = round(torch.cuda.memory_cached(i) / 1024**3, 1)
        print(f'GPU: {i} Allocated: {max_alloc}G Cached: {cached}G')


def load_header(conversation_dir):
    """[summary]
    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID
    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, 'misc')

    # TODO: Assuming name of mat file is the same conversation_dir
    # TODO: write a condition to check this
    header_file = glob.glob(os.path.join(misc_dir, '*_header.mat'))[0]
    if not os.path.exists(header_file):
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label
    return labels
