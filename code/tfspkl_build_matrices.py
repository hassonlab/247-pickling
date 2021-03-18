import glob
import os

import numpy as np
import pandas as pd
from electrode_utils import return_electrode_array
from tfspkl_utils import (extract_conversation_contents, get_common_electrodes,
                          get_conversation_list)


def function1(x):
    split_list = x.split('conversation1')
    electrode = split_list[-1].strip('_')
    subject = int(split_list[0][2:5])
    return (subject, electrode)


def build_design_matrices(CONFIG, delimiter=','):
    """Build examples and labels for the model

    Args:
        CONFIG (dict): configuration information
        delimiter (str, optional): conversation delimier. Defaults to ','.

    Returns:
        tuple: (signals, labels)

    Misc:
        signals: neural activity data
        labels: words/n-grams/sentences
    """
    if CONFIG['sig_elec_file']:
        try:
            bobbi = pd.read_csv(CONFIG['sig_elec_file'],
                                header=None)[0].tolist()
            sigelec_list = [function1(item) for item in bobbi]
            df = pd.DataFrame(sigelec_list, columns=['subject', 'electrode'])
        except:
            df = pd.read_csv(CONFIG['sig_elec_file'],
                             columns=['subject', 'electrode'])
        else:
            electrodes_dict = df.groupby('subject')['electrode'].apply(
                list).to_dict()

        full_signal = []
        trimmed_signal = []
        binned_signal = []
        electrode_names = []
        electrodes = []
        subject_id = []
        for subject, electrode_labels in electrodes_dict.items():
            (full_signal_part, full_stitch_index, trimmed_signal_part,
             trimmed_stitch_index, binned_signal_part, bin_stitch_index,
             all_examples, all_trimmed_examples, convo_all_examples_size,
             convo_trimmed_examples_size, electrodes_part,
             electrode_names_part, conversations,
             subject_id_part) = process_data_for_pickles(
                 CONFIG, subject, electrode_labels)

            full_signal.append(full_signal_part)
            trimmed_signal.append(trimmed_signal_part)
            binned_signal.append(binned_signal_part)

            electrode_names.extend(electrode_names_part)
            electrodes.extend(electrodes_part)
            subject_id.extend(subject_id_part)

        conversations = [None]

        full_signal = np.concatenate(full_signal, axis=1)
        trimmed_signal = np.concatenate(trimmed_signal, axis=1)
        binned_signal = np.concatenate(binned_signal, axis=1)

    else:
        (full_signal, full_stitch_index, trimmed_signal, trimmed_stitch_index,
         binned_signal, bin_stitch_index, all_examples, all_trimmed_examples,
         convo_all_examples_size, convo_trimmed_examples_size, electrodes,
         electrode_names, conversations,
         subject_id) = process_data_for_pickles(CONFIG)

    return (full_signal, full_stitch_index, trimmed_signal,
            trimmed_stitch_index, binned_signal, bin_stitch_index,
            all_examples, all_trimmed_examples, convo_all_examples_size,
            convo_trimmed_examples_size, electrodes, electrode_names,
            conversations, subject_id)


def process_data_for_pickles(CONFIG, subject=None, electrode_labels=None):
    exclude_words = CONFIG["exclude_words"]
    suffix = '/misc/*trimmed.txt'

    conversations = get_conversation_list(CONFIG, subject)
    electrodes, electrode_names = get_common_electrodes(CONFIG, conversations)

    if electrode_labels:
        idx = [
            i for i, e in enumerate(electrode_names) if e in electrode_labels
        ]
        electrode_names = [electrode_names[i] for i in idx]
        electrodes = [electrodes[i] for i in idx]

        assert set(electrode_names) == set(electrode_labels)

    if subject:
        subject_id = [subject for i in electrodes]
    else:
        subject_id = [CONFIG["subject"] for i in electrodes]

    full_signal, trimmed_signal, binned_signal = [], [], []
    full_stitch_index, trimmed_stitch_index, bin_stitch_index = [], [], []

    all_examples = []
    all_trimmed_examples = []

    convo_all_examples_size = []
    convo_trimmed_examples_size = []

    for conversation in conversations:
        try:  # Check if files exists
            datum_fn = glob.glob(conversation + suffix)[0]
        except IndexError:
            print('File DNE: ', conversation + suffix)
            continue

        # Extract electrode data (signal_length, num_electrodes)
        ecogs = return_electrode_array(CONFIG, conversation, electrodes)
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

        examples = extract_conversation_contents(datum_fn, exclude_words)

        cutoff_portion = signal_length % bin_size
        if cutoff_portion:
            ecogs = ecogs[:-cutoff_portion, :]
            signal_length = ecogs.shape[0]

        split_indices = np.arange(bin_size, signal_length, bin_size)
        convo_binned_signal = np.vsplit(ecogs, split_indices)

        # TODO: think about this line
        trimmed_examples = list(
            filter(lambda x: x[2] < signal_length, examples))
        convo_all_examples_size.append(len(examples))
        convo_trimmed_examples_size.append(len(trimmed_examples))

        trimmed_signal.append(ecogs)
        trimmed_stitch_index.append(signal_length)

        mean_binned_signal = [
            np.mean(split, axis=0) for split in convo_binned_signal
        ]

        mean_binned_signal = np.vstack(mean_binned_signal)
        bin_stitch_index.append(mean_binned_signal.shape[0])

        binned_signal.append(mean_binned_signal)

        all_examples.append(examples)
        all_trimmed_examples.append(trimmed_examples)

        print(os.path.basename(conversation), a, len(examples), ecogs.shape[0],
              len(trimmed_examples), mean_binned_signal.shape[0])

    full_signal = np.concatenate(full_signal)
    full_stitch_index = np.cumsum(full_stitch_index).tolist()

    trimmed_signal = np.concatenate(trimmed_signal)
    trimmed_stitch_index = np.cumsum(trimmed_stitch_index).tolist()

    binned_signal = np.vstack(binned_signal)
    bin_stitch_index = np.cumsum(bin_stitch_index).tolist()

    return (full_signal, full_stitch_index, trimmed_signal,
            trimmed_stitch_index, binned_signal, bin_stitch_index,
            all_examples, all_trimmed_examples, convo_all_examples_size,
            convo_trimmed_examples_size, electrodes, electrode_names,
            conversations, subject_id)
