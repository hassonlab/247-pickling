import gensim.downloader as api
import tfsemb_download as tfsemb_dwnld
import pandas as pd
import json
from tfsemb_config import setup_environ
from tfsemb_main import tokenize_and_explode
from tfsemb_parser import arg_parser
from utils import load_pickle, main_timer
from utils import save_pickle as svpkl


def add_vocab_columns(args, df, column=None):
    """Add columns to the dataframe indicating whether each word is in the
    vocabulary of the language models we're using.
    """

    # Add language models
    for model in [
        *tfsemb_dwnld.CAUSAL_MODELS,
        *tfsemb_dwnld.SEQ2SEQ_MODELS,
        *tfsemb_dwnld.SPEECHSEQ2SEQ_MODELS,
        *tfsemb_dwnld.MLM_MODELS,
    ]:
        try:
            tokenizer = tfsemb_dwnld.download_hf_tokenizer(model, local_files_only=True)
        except:
            tokenizer = tfsemb_dwnld.download_hf_tokenizer(
                model, local_files_only=False
            )

        key = tfsemb_dwnld.clean_lm_model_name(model)
        print(f"Adding column: (token) in_{key}")

        try:
            curr_vocab = tokenizer.vocab
        except AttributeError:
            curr_vocab = tokenizer.get_vocab()

        def helper(x):
            if len(tokenizer.tokenize(x)) == 1:
                return isinstance(curr_vocab.get(tokenizer.tokenize(x)[0]), int)
            return False

        df[f"in_{key}"] = df[column].apply(helper)

    return df


@main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    # base_df = load_pickle(args.labels_pickle, "labels")
    base_df = pd.read_csv(args.labels_pickle, index_col=0)
    base_df.insert(0, "word_idx", base_df.index.values)

    # filter out words
    filter_values = ["[inaudible]", "{inaudible}", "{inaudbile}", "{Gasps}", "{LG}"]
    base_df = base_df[~base_df.word.isin(filter_values)]

    with open(args.lag_json, "r") as j:
        lag_info = json.loads(j.read())

    base_df["adjusted_onset"] = base_df.start + lag_info["lag_s"]
    base_df["adjusted_offset"] = base_df.end + lag_info["lag_s"]

    # base_df.loc[:, "word"] = base_df.word.apply(  # HACK strip punc
    #     lambda x: x.translate(
    #         str.maketrans("", "", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    #     )
    # )

    glove = api.load("glove-wiki-gigaword-50")
    base_df["in_glove50"] = base_df.word.str.lower().apply(
        lambda x: isinstance(glove.key_to_index.get(x), int)
    )

    if args.embedding_type == "glove50":
        base_df = base_df[base_df["in_glove50"]]
        base_df = add_vocab_columns(args, base_df, column="word")
    else:
        # base_df = base_df[base_df.speaker.str.contains("Speaker")]  # HACK
        base_df = tokenize_and_explode(args, base_df)
        # base_df = add_vocab_columns(args, base_df, column="token2word")

    svpkl(base_df, args.base_df_file)


if __name__ == "__main__":
    main()
