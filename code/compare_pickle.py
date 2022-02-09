import os
import numpy as np
import pandas as pd
import pickle
import string
import glob


def load_pickle(emb_dir, emb_name):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    pickle_name = os.path.join(emb_dir, emb_name)
    with open(pickle_name, 'rb') as fh:
        emb = pickle.load(fh)

    return emb


def get_emb_dir(path, subject, emb_type, layer):
    return os.path.join(path, subject, 'embeddings', emb_type, 'full', layer)


def compare_emb(emb, df1, df2):
    print("Comparing Embedding ", emb)
    if len(df1) == len(df2):
        for idx,_ in enumerate(df1):
            if df1[idx]['adjusted_onset'] != df2[idx]['adjusted_onset']:
                print(df1[idx]['adjusted_onset'])
                print(df2[idx]['adjusted_onset'])
                return('Onset different')
            elif df1[idx]['adjusted_offset'] != df2[idx]['adjusted_offset']:
                return('Offset different')
            elif not np.array_equal(df1[idx]['embeddings'], df2[idx]['embeddings']):
                return('Embeddings different')
        return("Embedding is the same")
    else:
        return("Embedding not the same length")


def main():

    project_id = 'tfs'
    subject = '676'
    emb_type = 'glove50'
    layer = 'layer_01'
    RESULTS_DIR1 = os.path.join(os.getcwd(), 'results', project_id)
    RESULTS_DIR2 = os.path.join('/scratch/gpfs/zzada/247-pickling', 'results', project_id)

    EMB_DIR1 = get_emb_dir(RESULTS_DIR1, subject, emb_type, layer)
    EMB_DIR2 = get_emb_dir(RESULTS_DIR2, subject, emb_type, layer)

    emb_list1 = os.listdir(EMB_DIR1)
    emb_list2 = os.listdir(EMB_DIR2)
    
    dif_list = []
    for emb in emb_list1:
        df1 = load_pickle(EMB_DIR1, emb)
        df2 = load_pickle(EMB_DIR2, emb)
        compare_result = compare_emb(emb, df1, df2)
        if compare_result != "Embedding is the same":
            dif_list.append([emb,compare_result])

    print(dif_list)

    # PKL_DIR = os.path.join(RESULTS_DIR, subject, 'pickles')

    breakpoint()

    return 0

if __name__ == '__main__':
    main()