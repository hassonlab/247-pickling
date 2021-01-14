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


def return_examples_new(file, delim, ex_words, vocab_str='std'):
    df = pd.read_csv(file,
                     sep=' ',
                     header=None,
                     names=['word', 'onset', 'offset', 'accuracy', 'speaker'])
    df['word'] = df['word'].str.lower().str.strip()
    df = df[~df['word'].isin(ex_words)]

    return df.values.tolist()


def load_header(conversation_dir):
    """[summary]
    Args:
        conversation_dir ([type]): [description]
        subject_id (string): Subject ID
    Returns:
        list: labels
    """
    misc_dir = os.path.join(conversation_dir, 'misc')

    # TODO: Assuming name of mat file is the same as conversation_dir
    # TODO: write a condition to check this
    header_file = glob.glob(os.path.join(misc_dir, '*_header.mat'))[0]
    if not os.path.exists(header_file):
        return
    header = mat73.loadmat(header_file)
    labels = header.header.label
    return labels
