import os
import pickle
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold


# https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
def get_git_revision_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def load_pickle(pickle_name, key=None):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(pickle_name, "rb") as fh:
        datum = pickle.load(fh)

    if key:
        df = pd.DataFrame.from_dict(datum[key])
    else:
        df = pd.DataFrame.from_dict(datum)

    return df


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as fh:
        pickle.dump(item, fh)
    return


def get_current_time(timer_string=""):
    time_now = datetime.now()
    print(f'{timer_string} Time: {time_now.strftime("%A %m/%d/%Y %H:%M:%S")}')
    return time_now


def main_timer(func):
    def function_wrapper():
        start_time = get_current_time("Start")
        func()
        end_time = get_current_time("End")
        print(f"Total runtime: {end_time - start_time} (HH:MM:SS)")

    return function_wrapper


def lcs(x, y):
    """
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    """
    m = len(x)
    n = len(y)
    c = np.zeros((m + 1, n + 1), dtype=np.int)

    for i in tqdm.trange(1, m + 1, leave=True, desc="Aligning"):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i, j] = c[i - 1, j - 1] + 1
            else:
                c[i, j] = max(c[i, j - 1], c[i - 1, j])

    mask1, mask2 = [], []
    i = m
    j = n
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            i -= 1
            j -= 1
            mask1.append(i)
            mask2.append(j)

        elif c[i - 1][j] > c[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return mask1[::-1], mask2[::-1]


def stratify_split(df, num_folds, split_str=None):
    # Extract only test folds
    if split_str is None:
        skf = KFold(n_splits=num_folds, shuffle=False, random_state=0)
    elif split_str == "stratify":
        skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=0)
    else:
        raise Exception("wrong string")

    folds = [t[1] for t in skf.split(df, df.word)]
    return folds


def create_folds(df, num_folds, split_str=None):
    """create new columns in the df with the folds labeled

    Args:
        args (namespace): namespace object with input arguments
        df (DataFrame): labels
    """
    fold_column_names = ["fold" + str(i) for i in range(num_folds)]
    folds = stratify_split(df, num_folds, split_str=split_str)

    # Go through each fold, and split
    for i, fold_col in enumerate(fold_column_names):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds

        folds_ixs = np.roll(folds, i)
        *_, dev_ixs, test_ixs = folds_ixs

        df[fold_col] = "train"
        df.loc[dev_ixs, fold_col] = "dev"
        df.loc[test_ixs, fold_col] = "test"

    return df
