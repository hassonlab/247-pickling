import glob
import os

import torch


def setup_environ(args):

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
    args.conversation_list = sorted(glob.glob1(args.input_dir, "NY*Part*conversation*"))

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

    return
