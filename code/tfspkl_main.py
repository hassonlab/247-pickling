'''
Filename: /scratch/gpfs/hgazula/247-project/tfs_pickling.py
Path: /scratch/gpfs/hgazula/247-project
Created Date: Tuesday, December 1st 2020, 8:19:27 pm
Author: Harshvardhan Gazula
Description: Contains code to pickle 247 data

Copyright (c) 2020 Your Company
'''
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tfspkl_build_matrices import build_design_matrices
from tfspkl_config import build_config
from tfspkl_parser import arg_parser
from utils import main_timer


def save_pickle(args, item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = os.path.join(args.PKL_DIR, file_name) + add_ext

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def find_switch_points(array):
    """Find indices where speaker switches and split the dataframe
    """
    return np.where(array[:-1] != array[1:])[0] + 1


def get_sentence_length(section):
    """Sentence length = offset of the last word - onset of first word
    """
    last_word_offset = section.iloc[-1, 2]
    first_word_onset = section.iloc[0, 1]
    return last_word_offset - first_word_onset


def append_sentence(section):
    """Join the words to form a sentence and append

    Args:
        section (DataFrame): [description]
    """
    sentence = ' '.join(section['word'])
    section['sentence'] = sentence
    return section


def append_sentence_length(section):
    sentence_length = get_sentence_length(section)
    section['sentence_signal_length'] = sentence_length
    return section


def append_num_words(section):
    section['num_words'] = len(section)
    return section


def append_sentence_idx(section, idx):
    section['sentence_idx'] = idx + 1
    return section


def convert_labels_to_df(labels):
    convo_df = pd.DataFrame(
        labels, columns=['word', 'onset', 'offset', 'accuracy', 'speaker'])
    return convo_df


def split_convo_to_sections(conversation):
    convo_df = convert_labels_to_df(conversation)
    speaker_switch_idx = find_switch_points(convo_df.speaker.values)
    sentence_df = np.split(convo_df, speaker_switch_idx, axis=0)
    return sentence_df


def process_sections(section_list):
    # For each sentence df split
    my_labels = []
    for idx, section in enumerate(section_list):
        section = append_sentence_length(section)
        section = append_sentence(section)
        section = append_num_words(section)
        section = append_sentence_idx(section, idx)
        my_labels.append(section)
    return pd.concat(my_labels, ignore_index=True)


def create_sentence(conversation):
    """[summary]

    Args:
        labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    convo_sections = split_convo_to_sections(conversation)
    conversation = process_sections(convo_sections)
    return conversation


def word_stemming(conversation, ps):
    conversation['stemmed_word'] = conversation['word'].apply(ps.stem)
    return conversation


def shift_onsets(conversation, shift):
    conversation['adjusted_onset'] += shift
    conversation['adjusted_offset'] += shift
    return conversation


def add_sentence_index(conversation, length):
    conversation['sentence_idx'] += length
    length = conversation['sentence_idx'].nunique()
    return conversation, length


def add_conversation_id(conversation, conv_id):
    conversation['conversation_id'] = conv_id
    return conversation


def add_conversation_name(conversation, name):
    conversation['conversation_name'] = os.path.basename(name)
    return conversation


def process_labels(trimmed_stitch_index, labels, conversations):
    """Adjust label onsets to account for stitched signal length.
    Also peform stemming on the labels.

    Args:
        trimmed_stitch_index (list): stitch indices of trimmed signal
        labels (list): of tuples (word, speaker, onset, offset, accuracy)

    Returns:
        DataFrame: labels
    """
    trimmed_stitch_index.insert(0, 0)
    trimmed_stitch_index.pop(-1)

    new_labels = []

    len_to_add = 0
    for conv_id, (conversation_name, start, sub_list) in enumerate(
            zip(conversations, trimmed_stitch_index, labels), 1):

        sub_list = create_sentence(sub_list)
        sub_list = shift_onsets(sub_list, start)
        sub_list = add_conversation_id(sub_list, conv_id)
        sub_list = add_conversation_name(sub_list, conversation_name)
        sub_list, len_to_add = add_sentence_index(sub_list, len_to_add)

        new_labels.append(sub_list)

    return pd.concat(new_labels, ignore_index=True)


def inclass_word_freq(df):
    df['word_freq_phase'] = df.groupby(['word', 'production'
                                        ])['word'].transform('count')
    return df


def total_word_freq(df):
    df['word_freq_overall'] = df.groupby(['word'])['word'].transform('count')
    return df


def create_production_flag(df):
    df['production'] = (df['speaker'] == 'Speaker1').astype(int)
    return df


def filter_on_freq(args, df):
    df = df.groupby('word').filter(
        lambda x: len(x) >= args.vocab_min_freq).reset_index(drop=True)
    return df


def stratify_split(df, split_str=None):
    # Extract only test folds
    if split_str is None:
        skf = KFold(n_splits=5, shuffle=True, random_state=0)
    elif split_str == 'stratify':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    else:
        raise Exception('wrong string')

    folds = [t[1] for t in skf.split(df, df.word)]
    return folds


def create_folds(args, df, split_str=None):
    """create new columns in the df with the folds labeled

    Args:
        args (namespace): namespace object with input arguments
        df (DataFrame): labels
    """
    fold_column_names = ['fold' + str(i) for i in range(5)]
    folds = stratify_split(df, split_str=split_str)

    # Go through each fold, and split
    for i, fold_col in enumerate(fold_column_names):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds

        folds_ixs = np.roll(folds, i)
        *_, dev_ixs, test_ixs = folds_ixs

        df[fold_col] = 'train'
        df.loc[dev_ixs, fold_col] = 'dev'
        df.loc[test_ixs, fold_col] = 'test'

    return df


def create_labels_pickles(args,
                          stitch_index,
                          labels,
                          convo_labels_size,
                          convs,
                          label_str=None):
    labels_df = process_labels(stitch_index, labels, convs)
    labels_df = create_production_flag(labels_df)
    labels_df = inclass_word_freq(labels_df)
    labels_df = total_word_freq(labels_df)
    labels_df = create_folds(args, labels_df)

    labels_dict = dict(labels=labels_df.to_dict('records'),
                       convo_label_size=convo_labels_size)
    pkl_name = '_'.join([args.subject, label_str, 'labels'])
    save_pickle(args, labels_dict, pkl_name)

    if args.vocab_min_freq:
        labels_df = filter_on_freq(args, labels_df)
        labels_df = create_folds(args, labels_df, 'stratify')

        label_folds = labels_df.to_dict('records')
        pkl_name = '_'.join(
            [args.subject, label_str, 'labels_MWF',
             str(args.vocab_min_freq)])
        save_pickle(args, label_folds, pkl_name)


@main_timer
def main():
    args = arg_parser()
    args = build_config(args)

    (full_signal, full_stitch_index, trimmed_signal, trimmed_stitch_index,
     binned_signal, bin_stitch_index, full_labels, trimmed_labels,
     convo_full_examples_size, convo_trimmed_examples_size, electrodes,
     electrode_names, conversations) = build_design_matrices(vars(args),
                                                             delimiter=" ")

    # Create pickle with full signal
    full_signal_dict = dict(full_signal=full_signal,
                            full_stitch_index=full_stitch_index,
                            electrode_ids=electrodes,
                            electrode_names=electrode_names)
    save_pickle(args, full_signal_dict, args.subject + '_full_signal')

    # Create pickle with electrode maps
    electrode_map = dict(zip(electrodes, electrode_names))
    save_pickle(args, electrode_map, args.subject + '_electrode_names')

    # Create pickle with trimmed signal
    trimmed_signal_dict = dict(trimmed_signal=trimmed_signal,
                               trimmed_stitch_index=trimmed_stitch_index,
                               electrode_ids=electrodes,
                               electrode_names=electrode_names)
    save_pickle(args, trimmed_signal_dict, args.subject + '_trimmed_signal')

    # Create pickle with binned signal
    binned_signal_dict = dict(binned_signal=binned_signal,
                              bin_stitch_index=bin_stitch_index,
                              electrode_ids=electrodes,
                              electrode_names=electrode_names)
    save_pickle(args, binned_signal_dict, args.subject + '_binned_signal')

    # Create pickle with trimmed labels
    create_labels_pickles(args, trimmed_stitch_index, trimmed_labels,
                          convo_trimmed_examples_size, conversations,
                          'trimmed')
    create_labels_pickles(args, full_stitch_index, full_labels,
                          convo_full_examples_size, conversations, 'full')

    return


if __name__ == "__main__":
    main()
