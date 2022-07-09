import os
import torch
import torch.nn as nn


def setup_environ(args):

    DATA_DIR = os.path.join(os.getcwd(), "data", args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), "results", args.project_id)
    PKL_DIR = os.path.join(RESULTS_DIR, args.subject, "pickles")
    args.EMB_DIR = os.path.join(RESULTS_DIR, args.subject, "embeddings")

    args.full_model_name = args.embedding_type
    args.trimmed_model_name = args.embedding_type.split("/")[-1]

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_file = "_".join([args.subject, args.pkl_identifier, "labels.pkl"])
    args.pickle_name = os.path.join(PKL_DIR, labels_file)

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(os.listdir(args.input_dir))

    args.gpus = torch.cuda.device_count()

    stra = f"{args.trimmed_model_name}/cnxt_{args.context_length}"

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(
            args.EMB_DIR,
            args.pkl_identifier,
            stra,
            "layer_%02d",
        )
        output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir, output_file_name)

    return
