from email.mime import base
import os

from tfsemb_parser import arg_parser
from tfsemb_config import setup_environ
from utils import main_timer, load_pickle
from tfsemb_main import tokenize_and_explode
from utils import save_pickle as svpkl
import tfsemb_download as tfsemb_dwnld


def add_vocab_columns_new(args, df):
    """Add columns to the dataframe indicating whether each word is in the
    vocabulary of the language models we're using.
    """

    # Add glove
    # glove = api.load("glove-wiki-gigaword-50")
    # df["in_glove"] = df.word.str.lower().apply(
    #     lambda x: isinstance(glove.key_to_index.get(x), int)
    # )

    # Add language models
    for model in [*tfsemb_dwnld.CAUSAL_MODELS, *tfsemb_dwnld.SEQ2SEQ_MODELS]:
        if model != args.embedding_type:
            print(model)
            try:
                tokenizer = tfsemb_dwnld.download_hf_tokenizer(
                    model, local_files_only=True
                )
            except:
                tokenizer = tfsemb_dwnld.download_hf_tokenizer(
                    model, local_files_only=False
                )

            key = model.split("/")[-1]
            print(f"Adding column: (token) in_{key}_new")
            
            try:
                curr_vocab = tokenizer.vocab
            except AttributeError:
                curr_vocab = tokenizer.get_vocab()
            
            df[f"in_{key}_new"] = df.token.apply(
                lambda x: isinstance(curr_vocab.get(x), int)

            # TODO: remove in_{model} columns added in tfspkl_main.py
            # _new* added for testing purposes
    return df


@main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    if os.path.exists(args.base_df_file):
        print("Base DataFrame already exists")
        # Check if all in_{model} columns exists
    else:
        base_df = load_pickle(args.labels_pickle, "labels")
        if args.embedding_type != "glove50":
            base_df = tokenize_and_explode(args, base_df)
            base_df = add_vocab_columns_new(args, base_df)

        # TODO: add 'in_{model} columns here

        svpkl(base_df, args.base_df_file)


if __name__ == "__main__":
    main()
