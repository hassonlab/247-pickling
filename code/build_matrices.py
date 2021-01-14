import glob
import os

import numpy as np

from .electrode_utils import return_electrode_array
from .utils import return_conversations, return_examples_new


def build_design_matrices(CONFIG,
                          fs=512,
                          delimiter=',',
                          aug_shift_ms=[-500, -250, 250]):
    """Build examples and labels for the model

    Args:
        CONFIG (dict): configuration information
        fs (int, optional): frames per second. Defaults to 512.
        delimiter (str, optional): conversation delimier. Defaults to ','.
        aug_shift_ms (list, optional): shifts for data augmentation.
        Defaults to [-500, -250, 250].

    Returns:
        tuple: (signals, labels)

    Misc:
        signals: neural activity data
        labels: words/n-grams/sentences
    """
    exclude_words = CONFIG["exclude_words"]

    convs = return_conversations(CONFIG)
    cumsum_electrodes = list(np.cumsum(CONFIG['max_electrodes']))
    cumsum_electrodes.insert(0, 0)

    full_signal, trimmed_signal, binned_signal = [], [], []
    full_stitch_index, trimmed_stitch_index, bin_stitch_index = [], [], []
    all_examples = []
    convo_example_size = []
    for conversation, suffix, _, electrodes, electrode_names in convs:
        try:  # Check if files exists
            datum_fn = glob.glob(conversation + suffix)[0]
        except IndexError:
            print('File DNE: ', conversation + suffix)
            continue
        # Extract electrode data (signal_length, num_electrodes)
        ecogs = return_electrode_array(conversation, electrodes)
        if not ecogs.size:
            print(f'Bad Conversation: {conversation}')
            continue

        bin_size = 32  # 62.5 ms (62.5/1000 * 512)
        signal_length = ecogs.shape[0]

        if signal_length < bin_size:
            print("Ignoring conversation: Small signal")
            continue

        full_signal.append(ecogs)
        full_stitch_index.append(signal_length)
        a = ecogs.shape[0]

        examples = return_examples_new(datum_fn, delimiter, exclude_words,
                                       CONFIG["vocabulary"])
        b = len(examples)

        cutoff_portion = signal_length % bin_size
        if cutoff_portion:
            ecogs = ecogs[:-cutoff_portion, :]
            signal_length = ecogs.shape[0]

        split_indices = np.arange(bin_size, signal_length, bin_size)
        convo_binned_signal = np.vsplit(ecogs, split_indices)

        # TODO: think about this line
        # examples = list(filter(lambda x: x[2] < signal_length, examples))
        convo_example_size.append(len(examples))

        trimmed_signal.append(ecogs)
        trimmed_stitch_index.append(signal_length)

        mean_binned_signal = [
            np.mean(split, axis=0) for split in convo_binned_signal
        ]

        mean_binned_signal = np.vstack(mean_binned_signal)
        bin_stitch_index.append(mean_binned_signal.shape[0])

        binned_signal.append(mean_binned_signal)

        all_examples.append(examples)

        print(os.path.basename(conversation), a, b, ecogs.shape[0],
              len(examples), mean_binned_signal.shape[0])

    full_signal = np.concatenate(full_signal)
    full_stitch_index = np.cumsum(full_stitch_index).tolist()

    trimmed_signal = np.concatenate(trimmed_signal)
    trimmed_stitch_index = np.cumsum(trimmed_stitch_index).tolist()

    binned_signal = np.vstack(binned_signal)
    bin_stitch_index = np.cumsum(bin_stitch_index).tolist()

    return (full_signal, full_stitch_index, trimmed_signal,
            trimmed_stitch_index, binned_signal, bin_stitch_index,
            all_examples, convo_example_size, electrodes, electrode_names)
