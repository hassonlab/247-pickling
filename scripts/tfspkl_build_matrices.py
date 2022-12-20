import glob
import os

import numpy as np
import pandas as pd
from electrode_utils import return_electrode_array
from tfspkl_config import DATUM_FILE_MAP
from tfspkl_utils import (
    get_conversation_contents,
    get_all_electrodes,
    get_conversation_list,
)


def extract_subject_and_electrode(input_str):
    """Extract Subject and Electrode from the input string

    Args:
        input_str (str): {conversation_name}_{electrode_name}

    Returns:
        tuple: (subject, electrode)
    """
    split_list = input_str.split("conversation1")
    electrode = split_list[-1].strip("_")
    subject = int(split_list[0][2:5])
    return (subject, electrode)


def update_config(CONFIG, subject):
    CONFIG["subject"] = subject
    CONFIG["CONV_DIRS"] = os.path.join(CONFIG["DATA_DIR"], str(subject))
    return CONFIG


def build_design_matrices(CONFIG):
    """Build examples and labels for the model

    Args:
        CONFIG (dict): configuration information

    Returns:
        tuple: (signals, labels)

    Misc:
        signals: neural activity data
        labels: words/n-grams/sentences
    """
    if CONFIG["sig_elec_file"]:
        try:
            # If the electrode file is in Bobbi's original format
            sigelec_list = pd.read_csv(CONFIG["sig_elec_file"], header=None)[
                0
            ].tolist()
            sigelec_list = [
                extract_subject_and_electrode(item) for item in sigelec_list
            ]
            df = pd.DataFrame(sigelec_list, columns=["subject", "electrode"])
        except:
            # If the electrode file is in the new format
            df = pd.read_csv(
                CONFIG["sig_elec_file"],
                dtype={"subject": str, "electrode": str},
            )
        finally:
            electrodes_dict = (
                df.groupby("subject")["electrode"].apply(list).to_dict()
            )

        full_signal = []
        trimmed_signal = []
        binned_signal = []
        electrode_names = []
        electrodes = []
        subject_id = []
        for subject, electrode_labels in electrodes_dict.items():
            CONFIG = update_config(CONFIG, subject)
            (
                full_signal_part,
                full_stitch_index,
                trimmed_signal_part,
                trimmed_stitch_index,
                binned_signal_part,
                bin_stitch_index,
                all_examples,
                all_trimmed_examples,
                electrodes_part,
                electrode_names_part,
                conversations,
                subject_id_part,
            ) = process_data_for_pickles(CONFIG, electrode_labels)

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

        return (
            full_signal,
            full_stitch_index,
            trimmed_signal,
            trimmed_stitch_index,
            binned_signal,
            bin_stitch_index,
            all_examples,
            all_trimmed_examples,
            electrodes,
            electrode_names,
            conversations,
            subject_id,
        )

    else:
        return process_data_for_pickles(CONFIG)


def get_datum_suffix(CONFIG):
    """Return subject's corresponding datum file suffix"""
    datum_file_suffix = DATUM_FILE_MAP.get(CONFIG["project_id"], None).get(
        CONFIG["subject"], None
    )

    if not datum_file_suffix:
        print("Incorrect Project ID or Subject")
        exit()

    return datum_file_suffix


def process_data_for_pickles(CONFIG, electrode_labels=None):

    datum_file_suffix = get_datum_suffix(CONFIG)

    conversations = get_conversation_list(CONFIG)
    electrodes, electrode_names = get_all_electrodes(CONFIG, conversations)

    if electrode_labels:
        idx = [
            i for i, e in enumerate(electrode_names) if e in electrode_labels
        ]

        electrodes, electrode_names = zip(
            *[(electrodes[i], electrode_names[i]) for i in idx]
        )

        assert len(set(electrode_names) - set(electrode_labels)) == 0

    subject_id = [CONFIG["subject"] for i in electrodes]

    full_signal, trimmed_signal, binned_signal = [], [], []
    full_stitch_index, trimmed_stitch_index, bin_stitch_index = [], [], []

    all_examples = []
    all_trimmed_examples = []

    for conv_idx, conversation in enumerate(conversations, 1):
        try:  # Check if files exists
            datum_fn = glob.glob(
                os.path.join(conversation, "misc", datum_file_suffix)
            )[0]
        except IndexError:
            print(
                "File DNE: ",
                os.path.join(conversation, "misc", datum_file_suffix),
            )
            continue

        # Extract electrode data (signal_length, num_electrodes)
        ecogs = return_electrode_array(CONFIG, conversation, electrodes)
        if not ecogs.size:
            print(f"Bad Conversation: {conversation}")
            continue

        bin_size = 32  # 62.5 ms (62.5/1000 * 512)
        signal_length = ecogs.shape[0]

        if signal_length < bin_size:
            print("Ignoring conversation: Small signal")
            continue

        full_signal.append(ecogs)
        full_stitch_index.append(signal_length)
        a = ecogs.shape[0]

        examples_df = get_conversation_contents(CONFIG, datum_fn)

        # examples = examples_df.values.tolist()

        cutoff_portion = signal_length % bin_size
        if cutoff_portion:
            ecogs = ecogs[:-cutoff_portion, :]
            signal_length = ecogs.shape[0]

        split_indices = np.arange(bin_size, signal_length, bin_size)
        convo_binned_signal = np.vsplit(ecogs, split_indices)

        # TODO: think about this line
        # trimmed_examples = list(
        #     filter(lambda x: x[2] < signal_length, examples))
        trimmed_examples = examples_df[
            examples_df.offset.isnull() | examples_df.offset < signal_length
        ]

        trimmed_signal.append(ecogs)
        trimmed_stitch_index.append(signal_length)

        mean_binned_signal = [
            np.mean(split, axis=0) for split in convo_binned_signal
        ]

        mean_binned_signal = np.vstack(mean_binned_signal)
        bin_stitch_index.append(mean_binned_signal.shape[0])

        binned_signal.append(mean_binned_signal)

        all_examples.append(examples_df)
        all_trimmed_examples.append(trimmed_examples)

        print(
            f"{conv_idx:02d}",
            os.path.basename(conversation),
            a,
            len(examples_df),
            ecogs.shape[0],
            len(trimmed_examples),
            mean_binned_signal.shape[0],
        )

    full_signal = np.concatenate(full_signal)
    full_stitch_index = np.cumsum(full_stitch_index).tolist()

    trimmed_signal = np.concatenate(trimmed_signal)
    trimmed_stitch_index = np.cumsum(trimmed_stitch_index).tolist()

    binned_signal = np.vstack(binned_signal)
    bin_stitch_index = np.cumsum(bin_stitch_index).tolist()

    return (
        full_signal,
        full_stitch_index,
        trimmed_signal,
        trimmed_stitch_index,
        binned_signal,
        bin_stitch_index,
        all_examples,
        all_trimmed_examples,
        electrodes,
        electrode_names,
        conversations,
        subject_id,
    )
