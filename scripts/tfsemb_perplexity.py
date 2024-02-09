import os
import pickle
import sys

import gensim.downloader as api
import numpy as np
import pandas as pd
import tfsemb_download as tfsemb_dwnld
import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
from tfsemb_config import setup_environ
from tfsemb_parser import arg_parser
from utils import load_pickle, main_timer
from tqdm import tqdm


def select_conversation(args, df):
    if args.conversation_id:
        print("Selecting conversation", args.conversation_id)
        df = df[df.conversation_id == args.conversation_id]
    return df


def main():
    args = arg_parser()
    setup_environ(args)

    if os.path.exists(args.base_df_file):
        base_df = load_pickle(args.base_df_file)
    else:
        raise Exception("Base dataframe does not exist")

    # base_df_path = args.base_df_file.replace("661/embeddings", "777/pickles/embeddings")
    # base_df = load_pickle(base_df_path)

    utterance_df = select_conversation(args, base_df)
    assert len(utterance_df) != 0, "Empty dataframe"

    try:
        max_length = args.model.config.n_positions
    except:
        max_length = args.model.config.max_position_embeddings
    stride = 2048
    encodings = torch.tensor([tuple(utterance_df.token_id.tolist())])
    seq_len = encodings.size(1)

    nlls = []
    prev_end_loc = 0
    model = args.model
    device = args.device

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone().to(device)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            model = model.to(device)
            model.eval()
            outputs = args.model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Emb: {args.embedding_type}, Length: {max_length}, Perplexity: {ppl}")

    return


if __name__ == "__main__":
    main()