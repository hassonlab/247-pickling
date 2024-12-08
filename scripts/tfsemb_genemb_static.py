import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from tfsemb_config import setup_environ
from tfsemb_parser import arg_parser
from utils import load_pickle, main_timer
from utils import save_pickle as svpkl


def generate_llama3_static_embeddings(args, df):

    token_df = pd.read_pickle("llama3_dict.pkl")
    df1 = pd.DataFrame()
    df1["token_id"] = df.token_id
    df1 = df1.merge(token_df, how="left", on="token_id")
    df1 = df1.loc[:, ["embeddings"]]

    return df1


def main():
    args = arg_parser()
    setup_environ(args)
    file = f"results/tfs/%s/pickles/embeddings/Meta-Llama-3-8B/full/base_df.pkl"
    file2 = f"results/tfs/%s/embeddings/Meta-Llama-3-8B/full/base_df.pkl"

    all_df = pd.DataFrame()
    for sid in [625, 676, 7170, 798]:
        if str(sid)[0] == "6":
            base_df = load_pickle(file % sid)
        else:
            base_df = load_pickle(file2 % sid)
        all_df = pd.concat((all_df, base_df))

    token_df = pd.DataFrame({"token_id": sorted(all_df.token_id.unique())})
    tokens = torch.tensor([token_df.token_id.tolist()])
    embs = args.model.model.embed_tokens(tokens)
    token_df["embeddings"] = embs[0].tolist()
    breakpoint()
    token_df.to_pickle("llama3_dict.pkl")


if __name__ == "__main__":
    main()
