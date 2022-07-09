import argparse
import glob
import os
import shutil

import pandas as pd
from utils import load_pickle, save_pickle
from tfsemb_parser import arg_parser


def removeEmptyfolders(path):
    for (_path, _, _files) in os.walk(path, topdown=False):
        if _files:
            continue  # skip remove
        try:
            os.rmdir(_path)
            print("Remove :", _path)
        except OSError as ex:
            pass


def main():
    args = arg_parser()

    if args.subject == "625":
        num_convs = 54
    elif args.subject == "676" and "blenderbot" in args.embedding_type:
        num_convs = 76
    elif args.subject == "676":
        num_convs = 78
    else:
        num_convs = 1

    PKL_ROOT_DIR = os.path.join(
        os.getcwd(),
        "results",
        args.project_id,
        args.subject,
    )

    PKL_DIR = os.path.join(PKL_ROOT_DIR, "pickles")
    EMB_DIR = os.path.join(PKL_ROOT_DIR, "embeddings")

    trimmed_model_name = args.embedding_type.split("/")[-1]
    stra = f"{trimmed_model_name}/cnxt_{args.context_length}"
    args.output_dir = os.path.join(
        EMB_DIR,
        args.pkl_identifier,
        stra,
    )

    trimmed_labels = os.path.join(
        PKL_DIR,
        f"{args.subject}_trimmed_labels.pkl",
    )

    # copy base_df from source to target
    src = os.path.join(
        EMB_DIR, args.pkl_identifier, trimmed_model_name, "base_df.pkl"
    )
    dst = os.path.join(
        PKL_DIR,
        "embeddings",
        args.pkl_identifier,
        trimmed_model_name,
        "base_df.pkl",
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print("Base DataFrame Exists")
    else:
        print("Moving Base DataFrame")
        shutil.move(src, dst)

    layer_folders = sorted(os.listdir(args.output_dir))
    for layer_folder in layer_folders:
        print(f"Merging {layer_folder}")
        conversation_pickles = sorted(
            glob.glob(os.path.join(args.output_dir, layer_folder, "*"))
        )

        n = len(conversation_pickles)
        if n != num_convs:
            print(
                f"Bad conversation size: found {n} out of {num_convs}",
                f"in {args.output_dir}",
            )
            continue

        all_df = [
            load_pickle(os.path.join(args.output_dir, conversation))
            for conversation in conversation_pickles
        ]

        strb = "/".join([stra, layer_folder])
        args.emb_out_file = "/".join(
            ["embeddings", args.pkl_identifier, strb, "embeddings.pkl"]
        )
        all_df = pd.concat(all_df, ignore_index=True)

        all_exs = all_df.to_dict("records")
        full_emb_out_file = os.path.join(PKL_DIR, args.emb_out_file)
        os.makedirs(os.path.dirname(full_emb_out_file), exist_ok=True)
        save_pickle(all_exs, full_emb_out_file)

        if False:
            if "glove" in args.embedding_type or layer_folder in [
                "layer_48",
                "layer_16",
                "layer_8",
            ]:
                trimmed_df = load_pickle(trimmed_labels, key="labels")
                all_df.set_index(["conversation_id", "index"], inplace=True)
                trimmed_df.set_index(["conversation_id", "index"], inplace=True)
                all_df["adjusted_onset"] = None
                all_df["adjusted_offset"] = None
                all_df.update(trimmed_df)  # merge
                all_df.dropna(subset=["adjusted_onset"], inplace=True)
                all_df.reset_index(inplace=True)
                all_exs = all_df.to_dict("records")
                fn = args.emb_out_file.replace("full", "trimmed")
                save_pickle(all_exs, os.path.join(args.emb_out_dir, fn))

    # Deleting embeddings after concatenation
    shutil.rmtree(args.output_dir, ignore_errors=True)
    removeEmptyfolders(EMB_DIR)


if __name__ == "__main__":
    main()
