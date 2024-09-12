import gensim.downloader as api
import pandas as pd
import tfsemb_download as tfsemb_dwnld
from tfsemb_config import parse_arguments, setup_environ
from utils import load_pickle, main_timer
from utils import save_pickle as svpkl


def add_token_to_word(args, df):
    assert "token" in df.columns, "token column is missing"

    df["token2word"] = (
        df["token"]
        .apply(lambda x: [x])
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
        .str.lower()
    )
    return df


def add_token_id(args, df):
    df["token_id"] = df["token"].apply(args.tokenizer.convert_tokens_to_ids)
    return df


def add_token_is_root(args, df):
    token_is_root_string = args.trimmed_model_name + "_token_is_root"
    df[token_is_root_string] = (
        df["word"]
        == df["token"]
        .apply(lambda x: [x])
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
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
    df = add_token_to_word(args, df)
    df = add_token_id(args, df)
    df = add_token_is_root(args, df)

    df["token_idx"] = df.groupby(["adjusted_onset", "word"]).cumcount()
    df = df.reset_index(drop=True)

    return df


@main_timer
def main():
    args = parse_arguments()
    setup_environ(args, "tokenize")

    base_df = load_pickle(args.labels_pickle, "labels")

    # check if word in glove
    glove = api.load("glove-wiki-gigaword-50")
    base_df["in_glove50"] = base_df.word.str.lower().apply(
        lambda x: isinstance(glove.key_to_index.get(x), int)
    )

    if "glove" not in args.emb:  # tokenize
        base_df = tokenize_and_explode(args, base_df)

    svpkl(base_df, args.base_df_path, is_dataframe=True)


if __name__ == "__main__":
    main()
