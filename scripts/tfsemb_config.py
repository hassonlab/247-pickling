import glob
import os
import sys

import numpy as np
import tfsemb_download as tfsemb_dwnld
import torch


def get_model_layer_count(args):
    model = args.model
    max_layers = getattr(
        model.config,
        "n_layer",
        getattr(
            model.config,
            "num_layers",
            getattr(model.config, "num_hidden_layers", None),
        ),
    )

    # NOTE: layer_idx is shifted by 1 because the first item in hidden_states
    # corresponds to the output of the embeddings_layer
    if args.layer_idx == "all":
        args.layer_idx = np.arange(1, max_layers + 1)
    elif args.layer_idx == "last":
        args.layer_idx = [max_layers]
    else:
        layers = np.array(args.layer_idx)
        good = np.all((layers >= 0) & (layers <= max_layers))
        assert good, "Invalid layer number"

    return args


def select_tokenizer_and_model(args):

    model_name = args.full_model_name

    if model_name == "glove50":
        args.layer_idx = [1]
        return

    try:
        (
            args.model,
            args.tokenizer,
        ) = tfsemb_dwnld.download_tokenizers_and_models(
            model_name, local_files_only=True, debug=False
        )[
            model_name
        ]
    except OSError:
        # NOTE: Please refer to make-target: cache-models for more information.
        print(
            "Model and tokenizer not found. Please download into cache first.",
            file=sys.stderr,
        )
        return

    args = get_model_layer_count(args)

    if args.context_length <= 0:
        args.context_length = args.tokenizer.max_len_single_sentence

    assert (
        args.context_length <= args.tokenizer.max_len_single_sentence
    ), "given length is greater than max length"

    return


def process_inputs(args):
    if len(args.layer_idx) == 1:
        if args.layer_idx[0].isdecimal():
            args.layer_idx = int(args.layer_idx[0])
        else:
            args.layer_idx = args.layer_idx[0]
    else:
        try:
            args.layer_idx = list(map(int, args.layer_idx))
        except ValueError:
            print("Invalid layer index")
            exit(1)

    if args.embedding_type == "glove50":
        args.context_length = 1
        args.layer_idx = [1]

    return


def setup_environ(args):

    process_inputs(args)

    DATA_DIR = os.path.join(os.getcwd(), "data", args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), "results", args.project_id)

    args.PKL_DIR = os.path.join(RESULTS_DIR, args.subject, "pickles")
    args.EMB_DIR = os.path.join(RESULTS_DIR, args.subject, "embeddings")

    args.full_model_name = args.embedding_type
    args.trimmed_model_name = args.embedding_type.split("/")[-1]

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.labels_pickle = os.path.join(
        args.PKL_DIR,
        f"{args.subject}_{args.pkl_identifier}_labels.pkl",
    )

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(
        glob.glob1(args.input_dir, "NY*Part*conversation*")
    )

    select_tokenizer_and_model(args)
    stra = f"{args.trimmed_model_name}/{args.pkl_identifier}/cnxt_{args.context_length:04d}"

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(
            args.EMB_DIR,
            stra,
            "layer_%02d",
        )
        output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir, output_file_name)

        # saving the base dataframe
    args.base_df_file = os.path.join(
        args.EMB_DIR,
        args.trimmed_model_name,
        args.pkl_identifier,
        "base_df.pkl",
    )

    return
