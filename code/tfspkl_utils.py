import glob
import os

import mat73
import pandas as pd


def extract_elec_ids(conversation):
    """[summary]

    Args:
        conversation ([type]): [description]

    Returns:
        [type]: [description]
    """
    elec_files = glob.glob(os.path.join(conversation, 'preprocessed', '*.mat'))
    elec_ids_list = sorted(list(
        map(lambda x: int(os.path.splitext(x)[0].split('_')[-1]), elec_files)))

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
        elect_labels_list = extract_electrode_labels(conversation)

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
        list -- List of tuples (directory, file, idx, common_electrode_list)
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


def extract_conversation_contents(conversation, ex_words):
    """Return labels (lines) from conversation text

    Args:
        file ([type]): [description]
        ex_words ([type]): [description]

    Returns:
        list: list of lists with the following contents in that order
                ['word', 'onset', 'offset', 'accuracy', 'speaker']
    """
    df = pd.read_csv(conversation,
                     sep=' ',
                     header=None,
                     names=['word', 'onset', 'offset', 'accuracy', 'speaker'])
    df['word'] = df['word'].str.lower().str.strip()
    df = df[~df['word'].isin(ex_words)]

    return df.values.tolist()


def extract_electrode_labels(conversation_dir):
    """Read the header file electrode labels

    Args:
        conversation_dir (str): conversation folder name/path

    Returns:
        list: electrode labels
    """
    header_file = glob.glob(
        os.path.join(conversation_dir, 'misc', '*_header.mat'))[0]

    if not os.path.exists(header_file):
        return

    header = mat73.loadmat(header_file)
    labels = header.header.label

    return labels
