from datetime import datetime

import numpy as np
import tqdm


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
