from datetime import datetime

import numpy as np
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold


def main_timer(func):
    def function_wrapper():
        start_time = datetime.now()
        print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

        func()

        end_time = datetime.now()
        print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
        print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')

    return function_wrapper


def lcs(x, y):
    '''
    https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    '''
    m = len(x)
    n = len(y)
    c = np.zeros((m + 1, n + 1), dtype=np.int)

    for i in tqdm.trange(1, m + 1, leave=True, desc='Aligning'):
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
    elif split_str == 'stratify':
        skf = StratifiedKFold(n_splits=num_folds,
                              shuffle=False,
                              random_state=0)
    else:
        raise Exception('wrong string')

    folds = [t[1] for t in skf.split(df, df.word)]
    return folds


def create_folds(df, num_folds, split_str=None):
    """create new columns in the df with the folds labeled

    Args:
        args (namespace): namespace object with input arguments
        df (DataFrame): labels
    """
    fold_column_names = ['fold' + str(i) for i in range(5)]
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

        df[fold_col] = 'train'
        df.loc[dev_ixs, fold_col] = 'dev'
        df.loc[test_ixs, fold_col] = 'test'

    return df
