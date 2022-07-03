import argparse
import glob
import os

from utils import load_pickle, save_pickle


def split_embeddings(args, df):
    args.emb_out_file = "_".join(
        [args.subject, args.pkl_identifier, args.stra, "embeddings"]
    )

    filter_col = sorted([col for col in df if col.startswith("embeddings_layer_")])
    embeddings_df = df[filter_col]
    common_df = df.drop(filter_col, axis=1)

    for column in filter_col:
        common_df[column] = embeddings_df[column]
        all_df = all_df.to_dict("records")
        save_pickle(all_df, os.path.join(args.emb_out_dir, args.emb_out_file))

    pass


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="bert-large-uncased-whole-word-masking"
    )
    parser.add_argument("--embedding-type", type=str, default="glove")
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true", default=False)
    parser.add_argument("--save-hidden-states", action="store_true", default=False)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--subject", type=str, default="625")
    parser.add_argument("--history", action="store_true", default=False)
    parser.add_argument("--conversation-id", type=int, default=0)
    parser.add_argument("--pkl-identifier", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.subject == "625":
        num_convs = 54
    elif args.subject == "676":
        num_convs = 79
    else:
        num_convs = 1

    args.stra = "_".join([args.embedding_type, "cnxt", str(args.context_length)])
    args.output_dir = os.path.join(
        os.getcwd(),
        "results",
        args.project_id,
        args.subject,
        "embeddings_AllInOne",
        args.stra,
        args.pkl_identifier,
        "layer_00",
    )

    conversation_pickles = sorted(glob.glob(os.path.join(args.output_dir, "*.pkl")))
    assert len(conversation_pickles) == num_convs, "Bad conversation size"

    for idx, conversation in enumerate(conversation_pickles):
        conversation_name = os.path.split(conversation)[-1]
        print(conversation_name)
        conv_pkl = os.path.join(args.output_dir, conversation)
        df = load_pickle(conv_pkl)

        filter_col = sorted([col for col in df if col.startswith("embeddings_layer_")])
        embeddings_df = df[filter_col]
        df = df.drop(filter_col, axis=1)

        for idx, column in enumerate(filter_col, 1):
            output_dir = os.path.join(
                os.getcwd(),
                "results",
                args.project_id,
                args.subject,
                "embeddings_AllInOne",
                args.stra,
                args.pkl_identifier,
                f"layer_{idx:02}",
            )
            os.makedirs(output_dir, exist_ok=True)

            df["embeddings"] = embeddings_df[column]
            df.to_pickle(os.path.join(output_dir, conversation_name))

    os.rmdir(args.output_dir)


if __name__ == "__main__":
    main()
