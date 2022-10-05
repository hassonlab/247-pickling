import glob
import os
import shutil

import pandas as pd
from tfsemb_config import setup_environ
from tfsemb_parser import arg_parser
from tqdm import tqdm
from utils import load_pickle, save_pickle


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
    setup_environ(args)

    if args.subject == "625":
        num_convs = 54
    elif args.subject == "676" and "blenderbot" in args.embedding_type:
        num_convs = 76
    elif args.subject == "676":
        num_convs = 78
    elif args.subject == "7170":
        num_convs = 24
    else:
        num_convs = 1

    # stra = f"{args.trimmed_model_name}/{args.pkl_identifier}/cnxt_{args.context_length}"
    args.output_dir = os.path.join(
        args.EMB_DIR,
        args.trimmed_model_name,
        args.pkl_identifier,
        f"cnxt_{args.context_length:04d}",
    )

    # copy base_df from source to target
    src = os.path.join(os.path.dirname(args.output_dir), "base_df.pkl")
    dst = os.path.join(
        args.PKL_DIR,
        "embeddings",
        args.trimmed_model_name,
        args.pkl_identifier,
        "base_df.pkl",
    )
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print("Base DataFrame Exists")
    else:
        print("Moving Base DataFrame")
        shutil.move(src, dst)

    if not os.path.isdir(args.output_dir):
        print(f"DNE: {args.output_dir}")
        return
    else:
        layer_folders = sorted(os.listdir(args.output_dir))

    for layer_folder in tqdm(layer_folders, bar_format="Merging Layer..{n_fmt}"):
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

        all_df = pd.concat(all_df, ignore_index=True)
        all_exs = all_df.to_dict("records")

        args.emb_out_file = os.path.join(
            "embeddings",
            args.trimmed_model_name,
            args.pkl_identifier,
            f"cnxt_{args.context_length:04d}",
            layer_folder,
        )

        full_emb_out_file = os.path.join(args.PKL_DIR, args.emb_out_file)
        os.makedirs(os.path.dirname(full_emb_out_file), exist_ok=True)
        save_pickle(all_exs, full_emb_out_file)

        if False:
            if "glove" in args.embedding_type or layer_folder in [
                "layer_48",
                "layer_16",
                "layer_8",
            ]:
                # NOTE: args.trimmed_labels does not exist
                # check with ZZ about the utility of this code snippet
                trimmed_df = load_pickle(args.trimmed_labels, key="labels")
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
    removeEmptyfolders(args.EMB_DIR)


if __name__ == "__main__":
    main()
