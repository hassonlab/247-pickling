import glob
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat


def get_electrode(elec_id):
    """Extract neural data from mat files

    Arguments:
        elec_id (list): Electroide ID (subject specific electrodes)

    Returns:
        np.array: float32 numpy array of neural data
    """
    conversation, electrode = elec_id
    search_str = conversation + f'/preprocessed/*_{electrode}.mat'
    mat_fn = glob.glob(search_str)
    if len(mat_fn) == 0:
        print(f'[WARNING] electrode {electrode} DNE in {search_str}')
        return None
    return loadmat(mat_fn[0])['p1st'].squeeze().astype(np.float32)


def return_electrode_array(conv, elect):
    """Return neural data from all electrodes as a numpy object

    Arguments:
        conv (list): List of all conversations to be processed
        elect (list: int): List of electrode IDs to be processed

    Returns:
        Array -- Numpy object with neural data
    """
    elec_ids = ((conv, electrode) for electrode in elect)
    with Pool() as pool:
        ecogs = list(
            filter(lambda x: x is not None, pool.map(get_electrode, elec_ids)))

    ecogs = standardize_matrix(ecogs)
    assert (ecogs.ndim == 2 and ecogs.shape[1] == len(elect))

    return ecogs


def standardize_matrix(ecogs):
    ecogs = np.asarray(ecogs).T
    return (ecogs - np.mean(ecogs, axis=0)) / np.std(ecogs, axis=0)
