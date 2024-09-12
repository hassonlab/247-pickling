import os
import pickle
import sys

import numpy as np
import pandas as pd
import tfsemb_download as tfsemb_dwnld
import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
from tfsemb_config import setup_environ
from tfsemb_parser import arg_parser
from utils import load_pickle, main_timer
from utils import save_pickle as svpkl
from tfsemb_genemb_glove import generate_glove_embeddings
from tfsemb_genemb_causal import generate_causal_embeddings
from tfsemb_genemb_seq2seq import generate_conversational_embeddings
from tfsemb_genemb_whisper import generate_speech_embeddings
from tfsemb_genemb_mlm import generate_mlm_embeddings
from tfsemb_genemb_static import generate_llama3_static_embeddings


def save_pickle(args, item, embeddings=None):
    """Write 'item' to 'file_name.pkl'"""
    file_name = args.output_file
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    if embeddings is not None:
        for layer_idx, embedding in embeddings.items():
            item["embeddings"] = embedding.tolist()
            filename = file_name % layer_idx
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as fh:
                pickle.dump(item.to_dict("records"), fh)
    else:
        filename = file_name % args.layer_idx[0]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as fh:
            pickle.dump(item.to_dict("records"), fh)
    return


def select_conversation(args, df):
    if args.conversation_id:
        print("Selecting conversation", args.conversation_id)
        df = df[df.conversation_id == args.conversation_id]
    return df


def check_token_is_root(args, df):
    token_is_root_string = args.embedding_type.split("/")[-1] + "_token_is_root"
    df[token_is_root_string] = (
        df["word"]
        == df["token"]
        .apply(lambda x: [x])
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
    )
    return df


def convert_token_to_idx(args, df):
    df["token_id"] = df["token"].apply(args.tokenizer.convert_tokens_to_ids)
    return df


def convert_token_to_word(args, df):
    assert "token" in df.columns, "token column is missing"

    df["token2word"] = (
        df["token"]  # for gemma .apply(lambda x: [x])
        .apply(lambda x: [x])
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
        .str.lower()
    )
    return df


def tokenize_and_explode(args, df):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df["token"] = df.word.apply(args.tokenizer.tokenize)
    df = df.explode("token", ignore_index=False)
    df = convert_token_to_word(args, df)
    df = convert_token_to_idx(args, df)
    df = check_token_is_root(args, df)

    df["token_idx"] = df.groupby(["adjusted_onset", "word"]).cumcount()
    df = df.reset_index(drop=True)

    return df


# @main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    if os.path.exists(args.base_df_file):
        base_df = load_pickle(args.base_df_file)
    else:
        raise Exception("Base dataframe does not exist")

    utterance_df = select_conversation(args, base_df)
    print(
        args.conversation_id, utterance_df.conversation_name.unique(), len(utterance_df)
    )
    assert len(utterance_df) != 0, "Empty dataframe"

    # Select generation function based on model type
    match args.embedding_type:
        case "glove50":
            generate_func = generate_glove_embeddings
        case "Meta-Llama-3-8B-static":
            generate_func = generate_llama3_static_embeddings
        case item if item in tfsemb_dwnld.CAUSAL_MODELS:
            generate_func = generate_causal_embeddings
        case item if item in tfsemb_dwnld.SEQ2SEQ_MODELS:
            generate_func = generate_conversational_embeddings
        case item if item in tfsemb_dwnld.SPEECHSEQ2SEQ_MODELS:
            generate_func = generate_speech_embeddings
        case item if item in tfsemb_dwnld.MLM_MODELS:
            generate_func = generate_mlm_embeddings
        case _:
            print('Invalid embedding type: "{}"'.format(args.embedding_type))
            exit()

    # Generate Embeddings
    embeddings = None
    output = generate_func(args, utterance_df)
    if len(output) == 3:
        df, df_logits, embeddings = output
        if not df_logits.empty:
            svpkl(
                df_logits,
                os.path.join(args.logits_folder, args.output_file_name),
            )
    else:
        df = output

    save_pickle(args, df, embeddings)

    return


if __name__ == "__main__":
    # NOTE: Before running this script please refer to the cache-models target
    # in the Makefile
    main()
