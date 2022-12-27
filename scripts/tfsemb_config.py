import glob
import os
import sys

import numpy as np
import tfsemb_download as tfsemb_dwnld
import torch


def set_layer_idx(args):
    max_layers = tfsemb_dwnld.get_model_num_layers(args.embedding_type)

    # NOTE: layer_idx is shifted by 1 because the first item in hidden_states
    # corresponds to the output of the embeddings_layer
    match args.layer_idx:
        case "all":
            args.layer_idx = np.arange(0, max_layers + 1)
        case "last":
            args.layer_idx = [max_layers]
        case _:
            good = np.all((args.layers_idx >= 0) & (args.layers_idx <= max_layers))
            assert good, "Invalid layer number"


def set_context_length(args):
    if getattr(args, "tokenizer", None):
        max_context_length = args.tokenizer.max_len_single_sentence
    else:
        max_context_length = tfsemb_dwnld.get_max_context_length(args.embedding_type)

    if args.context_length <= 0:
        args.context_length = max_context_length

    assert (
        args.context_length <= max_context_length
    ), "given length is greater than max length"


def select_tokenizer_and_model(args):
    match args.embedding_type:
        case "glove50":
            args.context_length = 1
            args.layer_idx = [0]
        case item if item in [
            *tfsemb_dwnld.CAUSAL_MODELS,
            *tfsemb_dwnld.SEQ2SEQ_MODELS,
            *tfsemb_dwnld.MLM_MODELS,
        ]:
            (args.model, args.tokenizer,) = tfsemb_dwnld.download_tokenizers_and_models(
                item, local_files_only=True, debug=False
            )[item]
        case _:
            print(
                """Model and tokenizer not found. Please download into cache first.
                Please refer to make-target: cache-models for more information.""",
                file=sys.stderr,
            )
            exit()
    return


def process_inputs(args):
    if len(args.layer_idx) == 1:
        if isinstance(args.layer_idx[0], int) or args.layer_idx[0].isdecimal():
            args.layer_idx = [int(args.layer_idx[0])]
        else:
            args.layer_idx = args.layer_idx[0]
    else:
        try:
            args.layer_idx = list(map(int, args.layer_idx))
        except ValueError:
            print("Invalid layer index")
            exit(1)

    return


def setup_environ(args):

    select_tokenizer_and_model(args)
    process_inputs(args)
    if args.embedding_type != "glove50":
        set_layer_idx(args)
        set_context_length(args)

    DATA_DIR = os.path.join(os.getcwd(), "data", args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), "results", args.project_id)

    args.PKL_DIR = os.path.join(RESULTS_DIR, args.subject, "pickles")
    args.EMB_DIR = os.path.join(RESULTS_DIR, args.subject, "embeddings")

    args.trimmed_model_name = tfsemb_dwnld.clean_lm_model_name(args.embedding_type)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.labels_pickle = os.path.join(
        args.PKL_DIR,
        f"{args.subject}_{args.pkl_identifier}_labels.pkl",
    )

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(glob.glob1(args.input_dir, "NY*Part*conversation*"))

    stra = f"{args.trimmed_model_name}/{args.pkl_identifier}/cnxt_{args.context_length:04d}"

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(
            args.EMB_DIR,
            stra,
            "layer_%02d",
        )
        args.output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir, args.output_file_name)

    # saving the base dataframe
    args.base_df_file = os.path.join(
        args.EMB_DIR,
        args.trimmed_model_name,
        args.pkl_identifier,
        "base_df.pkl",
    )

    # saving logits as dataframe
    args.logits_folder = os.path.join(
        args.EMB_DIR,
        stra,
        "logits",
    )

    return
