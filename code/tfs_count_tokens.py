import glob
import os
import pickle

import numpy as np
import pandas as pd


def load_pickle(pickle_name):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum)

    return df


def process_df(df):
    df['is_nan'] = df['embeddings'].apply(lambda x: np.isnan(x).all())

    # drop empty embeddings
    df = df[~df['is_nan']]

    # use columns where token is root
    df = df[df['gpt2_token_is_root']]
    df = df[~df['glove50_embeddings'].isna()]

    return df


if __name__ == '__main__':
    conv_emb_dir = os.path.join(os.getcwd(), 'results', '625',
                                'conv_embeddings', '*')

    conv_list = sorted(glob.glob(conv_emb_dir))

    for i, conv in enumerate(conv_list, 1):
        df = load_pickle(conv)
        df = process_df(df)
        print(f'{i:02d}: {df.shape[0]}')
