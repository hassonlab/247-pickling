import numpy as np
import pandas as pd
import gensim.downloader as api


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def generate_glove_embeddings(args, df):
    # TODO implement new glove (issue 157)

    df1 = pd.DataFrame()
    glove = api.load("glove-wiki-gigaword-50")
    df1["embeddings"] = df["word"].apply(lambda x: get_vector(x.lower(), glove))

    return df1
