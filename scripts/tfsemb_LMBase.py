import gensim.downloader as api
import numpy as np
import tfsemb_download as tfsemb_dwnld
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


def get_windows(df):
    df.dropna(
        subset=["onset", "offset", "adjusted_onset", "adjusted_offset"],
        inplace=True,
    )

    # get datum of utterances
    df["new_conv"] = np.where(  # HACK ask Bobbi why
        (df.adjusted_onset - df.adjusted_offset.shift()) > 300 * 512, 1, 0
    )
    print(f"Word gap longer than 5 min for {df.new_conv.sum()} instances")
    df["utt_idx"] = (
        df.speaker.ne(df.speaker.shift())
        | df.conversation_id.ne(df.conversation_id.shift())
        | df.new_conv
    ).cumsum()

    df["utt_adjusted_onset"] = (
        df.loc[:, ("utt_idx", "adjusted_onset")].groupby("utt_idx").transform(min)
    )
    df["utt_adjusted_offset"] = (
        df.loc[:, ("utt_idx", "adjusted_offset")].groupby("utt_idx").transform(max)
    )
    df["adj_len"] = (df.adjusted_onset - df.onset).round(0).astype(int)
    df["utt_len"] = (df.utt_adjusted_offset - df.utt_adjusted_onset) / 512

    if sum(df.utt_len <= 0):  # filter non-positive utts
        len_df = len(df)
        df = df[df.utt_len > 0]
        print(f"Decreasing df from {len_df} to {len(df)}")

    keep_cols = [
        "speaker",
        "production",
        "conversation_name",
        "conversation_id",
        "utt_idx",
        "utt_adjusted_onset",
        "utt_adjusted_offset",
        "utt_len",
        "adj_len",
    ]
    df = df.loc[:, keep_cols]
    df = df.drop_duplicates(subset="utt_idx")  # get datum of utts

    # get datum of windows
    def get_window_num(x):  # get number of windows given utterance
        return int((x - 0.0325) // 0.02) + 1

    def get_windows(x):  # get windows to explode
        return np.arange(0, x)

    df["window_num"] = df.utt_len.apply(get_window_num)
    df.loc[df.window_num < 1, "window_num"] = 1  # at least 1
    df["windows"] = df.window_num.apply(get_windows)
    df = df.explode("windows", ignore_index=True)

    # for utts longer than 30s, split to chunks, shift onset offset and window_idx
    df["chunk_idx"] = df.windows // 1500  # chunk idx inside utt
    df["window_idx"] = df.windows % 1500  # window idx inside chunk
    df["utt_adjusted_onset"] = (
        df.utt_adjusted_onset + 30 * 512 * df.chunk_idx
    )  # shift utt onset
    df.loc[
        df.utt_adjusted_offset - df.utt_adjusted_onset > 30 * 512,
        "utt_adjusted_offset",
    ] = (
        df.utt_adjusted_onset + 30 * 512
    )  # shift utt offset
    df.loc[
        df.utt_adjusted_offset - df.utt_adjusted_onset == 30 * 512, "window_num"
    ] = 1500  # full window_nums
    df.window_num = (df.window_num - 1) % 1500 + 1  # shift window_num

    # for each window, shift onset and offset
    df["window_adjust"] = (df.window_idx - 2) * 0.02 + 0.0075
    df["adjusted_onset"] = df.utt_adjusted_onset + df.window_adjust * 512
    df["adjusted_offset"] = df.adjusted_onset + 0.065 * 512
    df.loc[
        df.adjusted_onset < df.utt_adjusted_onset, "adjusted_onset"
    ] = df.utt_adjusted_onset  # deal with chunk front
    df.loc[
        df.adjusted_offset > df.utt_adjusted_offset, "adjusted_offset"
    ] = df.utt_adjusted_offset  # deal with chunk back
    df["full_window"] = np.where(
        df.adjusted_offset - df.adjusted_onset == 0.065 * 512, 1, 0
    )

    # get on/offsets back from adjusted on/offsets
    df["utt_onset"] = df.utt_adjusted_onset - df.adj_len
    df["utt_offset"] = df.utt_adjusted_offset - df.adj_len
    df["onset"] = df.adjusted_onset - df.adj_len
    df["offset"] = df.adjusted_offset - df.adj_len

    keep_cols = [
        "speaker",
        "production",
        "conversation_name",
        "conversation_id",
        "utt_idx",
        "chunk_idx",
        "window_idx",
        "window_num",
        "utt_onset",
        "utt_offset",
        "utt_adjusted_onset",
        "utt_adjusted_offset",
        "onset",
        "offset",
        "adjusted_onset",
        "adjusted_offset",
        "full_window",
    ]
    df = df.loc[:, keep_cols]

    return df


@main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    base_df = load_pickle(args.labels_pickle, "labels")

    en_win = True  # HACK to get encoder windows
    if en_win:
        base_df = get_windows(base_df)
        print(len(base_df))
        return

    glove = api.load("glove-wiki-gigaword-50")
    base_df["in_glove50"] = base_df.word.str.lower().apply(
        lambda x: isinstance(glove.key_to_index.get(x), int)
    )

    if args.embedding_type == "glove50":
        base_df = base_df[base_df["in_glove50"]]
        base_df = add_vocab_columns(args, base_df, column="word")
    else:
        base_df = tokenize_and_explode(args, base_df)
        base_df = add_vocab_columns(args, base_df, column="token2word")

    svpkl(base_df, args.base_df_file)


if __name__ == "__main__":
    main()
