"""
Filename: /scratch/gpfs/hgazula/247-project/tfs_pickling.py
Path: /scratch/gpfs/hgazula/247-project
Created Date: Tuesday, December 1st 2020, 8:19:27 pm
Author: Harshvardhan Gazula
Description: Contains code to pickle 247 data

Copyright (c) 2020 Your Company
"""
import os

import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer as ps
from nltk.stem import WordNetLemmatizer as lt
from tfspkl_build_matrices import build_design_matrices
from tfspkl_config import build_config
from tfspkl_parser import arg_parser
from utils import main_timer, save_pickle

nltk.download("omw-1.4")


def find_switch_points(array):
    """Find indices where speaker switches and split the dataframe"""
    return np.where(array[:-1] != array[1:])[0] + 1


def get_sentence_length(section):
    """Sentence length = offset of the last word - onset of first word"""
    last_word_offset = section.offset.loc[section.offset.last_valid_index()]
    first_word_onset = section.onset.loc[section.onset.first_valid_index()]
    return last_word_offset - first_word_onset


def append_sentence(args, section):
    """Join the words to form a sentence and append

    Args:
        section (DataFrame): [description]
    """
    if args.project_id == "tfs":
        sentence = " ".join(section["word"])
        section["sentence"] = sentence
    else:
        section["sentence"] = None
    return section


def append_sentence_length(section):
    sentence_length = get_sentence_length(section)
    section["sentence_signal_length"] = sentence_length
    return section


def append_num_words(section):
    section["num_words"] = len(section)
    return section


def append_sentence_idx(section, idx):
    section["sentence_idx"] = idx + 1
    return section


def convert_labels_to_df(labels):
    convo_df = pd.DataFrame(
        labels, columns=["word", "onset", "offset", "accuracy", "speaker"]
    )
    return convo_df


def split_convo_to_sections(conversation):
    # convo_df = convert_labels_to_df(conversation)
    convo_df = conversation
    speaker_switch_idx = find_switch_points(convo_df.speaker.values)
    sentence_df = np.split(convo_df, speaker_switch_idx, axis=0)
    return sentence_df


def process_sections(args, section_list):
    # For each sentence df split
    my_labels = []
    for idx, section in enumerate(section_list):
        section = append_sentence_length(section)
        section = append_sentence(args, section)
        section = append_num_words(section)
        section = append_sentence_idx(section, idx)
        my_labels.append(section)
    return pd.concat(my_labels, ignore_index=True)


def create_sentence(args, conversation):
    """[summary]

    Args:
        labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    convo_sections = split_convo_to_sections(conversation)
    conversation = process_sections(args, convo_sections)
    return conversation


def word_stemming(conversation, ps):
    conversation["stemmed_word"] = conversation["word"].apply(ps.stem)
    return conversation


def shift_onsets(conversation, shift):
    conversation["adjusted_onset"] = conversation["onset"] + shift
    conversation["adjusted_offset"] = conversation["offset"] + shift
    return conversation


def add_sentence_index(conversation, length):
    conversation["sentence_idx"] += length
    length = conversation["sentence_idx"].nunique()
    return conversation, length


def add_conversation_id(conversation, conv_id):
    conversation["conversation_id"] = conv_id
    return conversation


def add_conversation_name(args, conversation, name):
    if args.project_id == "tfs":
        conversation["conversation_name"] = os.path.basename(name)
    else:
        conversation["conversation_name"] = None
    return conversation


def process_labels(args, stitch_index, labels, conversations):
    """Adjust label onsets to account for stitched signal length.
    Also perform stemming on the labels.

    Args:
        trimmed_stitch_index (list): stitch indices of trimmed signal
        labels (list): of tuples (word, speaker, onset, offset, accuracy)

    Returns:
        DataFrame: labels
    """
    stitch_index.insert(0, 0)
    stitch_index.pop(-1)

    new_labels = []

    len_to_add = 0
    for conv_id, (conversation_name, start, sub_list) in enumerate(
        zip(conversations, stitch_index, labels), 1
    ):

        sub_list = create_sentence(args, sub_list)
        sub_list = shift_onsets(sub_list, start)
        sub_list = add_conversation_id(sub_list, conv_id)
        sub_list = add_conversation_name(args, sub_list, conversation_name)
        sub_list, len_to_add = add_sentence_index(sub_list, len_to_add)

        new_labels.append(sub_list)

    return pd.concat(new_labels, ignore_index=True)


def add_word_freqs(df):
    grouped = df.word.str.lower().to_frame().groupby("word")
    df["word_freq_overall"] = grouped.word.transform("count")

    first = df[["word", "production"]].applymap(
        lambda x: x.lower() if type(x) == str else x
    )
    grouped = first.groupby(["word", "production"])
    df["word_freq_phase"] = grouped.word.transform("count")
    return df


def create_production_flag(df):
    """Create 'production' column with True or False

    Args:
        df (dataframe): label dataframe

    Returns:
        [dataframe]: same dataframe with a new column
    """
    df["production"] = (df["speaker"] == "Speaker1").astype(int)
    return df


def filter_on_freq(args, df):
    df = (
        df.groupby("word")
        .filter(lambda x: len(x) >= args.vocab_min_freq)
        .reset_index(drop=True)
    )
    return df


def calc_tokenizer_length(tokenizer, word):
    return False if pd.isnull(word) else len(tokenizer.tokenize(word)) == 1


def apply_lemmatize(word):
    return None if pd.isnull(word) else lt().lemmatize(word)


def apply_stemming(word):
    return None if pd.isnull(word) else ps().stem(word)


def add_lemmatize_stemming(df):
    df["lemmatized_word"] = df.word.str.strip().apply(
        lambda x: apply_lemmatize(x)
    )
    df["stemmed_word"] = df.word.str.strip().apply(lambda x: apply_stemming(x))

    return df


def add_fine_flag(args, df):
    """Add flag specifying whether the conversation is crude (0) or fine (1)
    Args:
        args (ArgParse): configuration object for the project
        df (DataFrame): labels/datum (with other columns added)
    Returns:
        DataFrame: df with a fine_flag column added
    """
    if args.crude_flag_file:
        flag_df = pd.read_csv(
            args.crude_flag_file,
            header=0,
            names=["conversation_name", "fine_flag"],
        )
        df = df.merge(flag_df, on="conversation_name")
    return df


def add_signal_length(df, stitch_index):
    """Add (full/trimmed) signal lengths to datum
    Args:
        df (DataFrame): datum being processed
        stitch_index (List): list of signal lengths for each conversation
    Returns:
        DataFrame: df with full and trimmed conversation signal length
    """
    df["full_signal_length"] = df["conversation_id"].map(
        dict(zip(df.conversation_id.unique(), stitch_index))
    )

    trim_stitch_index = np.array(stitch_index) - np.array(stitch_index) % 32
    df["trimmed_signal_length"] = df["conversation_id"].map(
        dict(zip(df.conversation_id.unique(), trim_stitch_index))
    )

    return df


def create_labels_pickles(args, stitch_index, labels, convs, label_str=None):
    labels_df = process_labels(args, stitch_index.copy(), labels, convs)
    labels_df = create_production_flag(labels_df)
    labels_df = add_word_freqs(labels_df)
    labels_df = add_lemmatize_stemming(labels_df)
    labels_df = add_fine_flag(args, labels_df)
    labels_df = add_signal_length(labels_df, stitch_index)

    labels_dict = dict(labels=labels_df.to_dict("records"))
    pkl_name = "_".join([args.subject, label_str, "labels"])
    pkl_name = os.path.join(args.PKL_DIR, pkl_name)
    save_pickle(labels_dict, pkl_name)


@main_timer
def main():
    # Read commandline arguments
    args = arg_parser()

    # Build variables needed for the project
    args = build_config(args)

    # Return signals and labels from *.mat and conversation.txt files
    (
        full_signal,
        full_stitch_index,
        trimmed_signal,
        trimmed_stitch_index,
        binned_signal,
        bin_stitch_index,
        full_labels,
        trimmed_labels,
        electrodes,
        electrode_names,
        conversations,
        subject_id,
    ) = build_design_matrices(dict(vars(args)))

    # Create pickle with full signal
    full_signal_dict = dict(
        full_signal=full_signal,
        full_stitch_index=full_stitch_index,
        electrode_ids=electrodes,
        electrode_names=electrode_names,
        subject=subject_id,
    )
    save_pickle(
        full_signal_dict,
        os.path.join(args.PKL_DIR, args.subject + "_full_signal"),
    )

    # Create pickle with full stitch index
    save_pickle(
        full_stitch_index,
        os.path.join(args.PKL_DIR, args.subject + "_full_stitch_index"),
    )

    # Create pickle with electrode maps
    electrode_map = dict(
        subject=subject_id,
        electrode_id=electrodes,
        electrode_name=electrode_names,
    )
    save_pickle(
        electrode_map,
        os.path.join(args.PKL_DIR, args.subject + "_electrode_names"),
    )

    # Create pickle with trimmed signal
    trimmed_signal_dict = dict(
        trimmed_signal=trimmed_signal,
        trimmed_stitch_index=trimmed_stitch_index,
        electrode_ids=electrodes,
        electrode_names=electrode_names,
        subject=subject_id,
    )
    save_pickle(
        trimmed_signal_dict,
        os.path.join(args.PKL_DIR, args.subject + "_trimmed_signal"),
    )

    # Create pickle with full stitch index
    save_pickle(
        trimmed_stitch_index,
        os.path.join(args.PKL_DIR, args.subject + "_trimmed_stitch_index"),
    )

    # Create pickle with binned signal
    binned_signal_dict = dict(
        binned_signal=binned_signal,
        bin_stitch_index=bin_stitch_index,
        electrode_ids=electrodes,
        electrode_names=electrode_names,
        subject=subject_id,
    )
    save_pickle(
        binned_signal_dict,
        os.path.join(args.PKL_DIR, args.subject + "_binned_signal"),
    )

    # Create pickle with full stitch index
    save_pickle(
        bin_stitch_index,
        os.path.join(args.PKL_DIR, args.subject + "_bin_stitch_index"),
    )

    # Create pickle with trimmed labels
    create_labels_pickles(
        args,
        trimmed_stitch_index,
        trimmed_labels,
        conversations,
        "trimmed",
    )
    print("SUCCESS: Trimmed Labels Pickle")

    create_labels_pickles(
        args,
        full_stitch_index,
        full_labels,
        conversations,
        "full",
    )
    print("SUCCESS: Full Labels Pickle")

    return


if __name__ == "__main__":
    main()
