import glob
import os
import sys
import yaml
import argparse
import getpass
import subprocess
import numpy as np
import tfsemb_download as tfsemb_dwnld
import tfspkl_utils
import torch
from utils import get_git_hash


def set_layer_idx(args):
    max_layers = tfsemb_dwnld.get_model_num_layers(args.emb)
    match args.layer_idx:
        case item if isinstance(item, np.ndarray):
            good = np.all([layer >= 0 for layer in args.layer_idx]) & np.all(
                [layer <= max_layers for layer in args.layer_idx]
            )
            assert good, "Invalid layer number"
        case "all":
            args.layer_idx = np.arange(0, max_layers + 1)
        case "last":
            args.layer_idx = [max_layers]


def set_context_length(args):
    if getattr(args, "tokenizer", None):
        max_context_length = max(
            args.tokenizer.max_len_single_sentence, args.tokenizer.model_max_length
        )
    else:
        max_context_length = tfsemb_dwnld.get_max_context_length(args.emb)

    if args.context_length <= 0:
        args.context_length = max_context_length

    assert (
        args.context_length <= max_context_length
    ), f"given length is greater than max length {max_context_length}"


def select_tokenizer_and_model(args, step):
    match args.emb:
        case "glove50":
            pass
        case item if item in [
            *tfsemb_dwnld.CAUSAL_MODELS,
            *tfsemb_dwnld.SEQ2SEQ_MODELS,
            *tfsemb_dwnld.MLM_MODELS,
            *tfsemb_dwnld.SPEECHSEQ2SEQ_MODELS,
        ]:
            (
                args.model,
                args.tokenizer,
                args.processor,
            ) = tfsemb_dwnld.download_tokenizers_and_models(
                step, item, local_files_only=True, debug=False
            )[
                item
            ]
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


def parse_arguments():
    """Read arguments from yaml config file

    Returns:
        namespace: all arguments from yaml config file
    """
    # parse yaml config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", nargs="*", type=str, default="[config.yml]")
    parser.add_argument("--conv-id", nargs="?", type=int, required=False, default=1)
    args = parser.parse_args()

    all_yml_args = {}
    for config_file in args.config_file:
        with open(f"configs/{config_file}", "r") as file:
            yml_args = yaml.safe_load(file)
            all_yml_args = all_yml_args | yml_args

    # get username
    user_id = getpass.getuser()
    all_yml_args["user_id"] = user_id
    all_yml_args["git_hash"] = get_git_hash()
    all_yml_args["conv_id"] = args.conv_id
    args = argparse.Namespace(**all_yml_args)
    try:  # eval lists
        args.layer_idx = eval(args.layer_idx)
    except:
        print("List parameter failed to eval")
    return args


def setup_environ(args, step):

    # Set up model, tokenizer, processor
    select_tokenizer_and_model(args, step)
    if step == "gen-emb":  # generating embeddings
        if args.emb == "glove50":
            args.context_length = 1
            args.layer_idx = [0]
        else:
            set_layer_idx(args)
            set_context_length(args)

    args.trimmed_model_name = tfsemb_dwnld.clean_lm_model_name(args.emb)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input directory paths (pickles)
    RESULTS_DIR = os.path.join(os.getcwd(), "results", args.project_id, str(args.sid))
    PKL_DIR = os.path.join(RESULTS_DIR, "pickles")
    args.labels_pickle = os.path.join(PKL_DIR, f"{args.sid}_full_labels.pkl")

    # output directory paths for tokenize (base df)
    MODEL_DIR = os.path.join(RESULTS_DIR, "embeddings", args.trimmed_model_name, "full")
    os.makedirs(MODEL_DIR, exist_ok=True)
    args.base_df_path = os.path.join(MODEL_DIR, "base_df.pkl")

    # output directory paths for gen-emb (emb df)
    if step == "gen-emb":
        DATA_DIR = os.path.join(os.getcwd(), "data", args.project_id, str(args.sid))
        conversation_lists = sorted(
            glob.glob1(DATA_DIR, "NY*Part*conversation*"),
            key=tfspkl_utils.custom_sort,
        )
        output_file_name = conversation_lists[args.conv_id - 1]
        EMB_DIR = os.path.join(MODEL_DIR, f"cnxt_{args.context_length:04d}")
        os.makedirs(EMB_DIR, exist_ok=True)
        args.emb_df_path = os.path.join(
            EMB_DIR, "layer_%02d", f"{output_file_name}.pkl"
        )
        args.logits_path = os.path.join(EMB_DIR, "logits", f"{output_file_name}.pkl")

    return
