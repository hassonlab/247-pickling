import glob
import sys
import warnings
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat


def get_electrode(CONFIG, elec_id):
    """Extract neural data from mat files

    Arguments:
        elec_id (list): Electroide ID (subject specific electrodes)

    Returns:
        np.array: float32 numpy array of neural data
    """
    conversation, electrode = elec_id

    if CONFIG["project_id"] == "podcast":
        search_str = conversation + f"/preprocessed_all/*_{electrode}.mat"
    elif CONFIG["project_id"] == "tfs":
        if CONFIG["subject"] == "7170":
            search_str = conversation + f"/preprocessed_v2/*_{electrode}.mat"
            # TODO: check if it is preprocessed or preprocessed_v2
        else:
            search_str = conversation + f"/preprocessed/*_{electrode}.mat"
    else:
        print("Incorrect Project ID")
        sys.exit()

    mat_fn = glob.glob(search_str)
    if mat_fn:
        return loadmat(mat_fn[0])["p1st"].squeeze().astype(np.float32)
    else:
        return None


def get_electrode_mp(elec_id, CONFIG):
    return get_electrode(CONFIG, elec_id)


def return_electrode_array(CONFIG, conv, elect):
    """Return neural data from all electrodes as a numpy object

    Arguments:
        conv (list): List of all conversations to be processed
        elect (list: int): List of electrode IDs to be processed

    Returns:
        Array -- Numpy object with neural data
    """
    elec_ids = ((conv, electrode) for electrode in elect)
    with Pool() as pool:
        ecogs = pool.map(partial(get_electrode_mp, CONFIG=CONFIG), elec_ids)

    ecogs = put_signals_into_array(ecogs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ecogs = standardize_matrix(ecogs)

    assert ecogs.ndim == 2 and ecogs.shape[1] == len(elect)

    return ecogs


def standardize_matrix(ecogs):
    ecogs = np.asarray(ecogs).T
    return (ecogs - np.mean(ecogs, axis=0)) / np.std(ecogs, axis=0)


def pad_bad_electrodes(item):
    # TODO: finish this function
    # load the subject_electrodes.pkl"
    # get the index of the bad electrode
    # print the electrode name and the amount of signal missing
    # print the conversation as well
    pass


def put_signals_into_array(ecogs):

    max_signal_length = max([item.size for item in ecogs if item is not None])
    ecogs = [
        np.repeat(np.nan, max_signal_length)
        if item is None
        else np.pad(item, (0, max_signal_length - len(item)))
        for item in ecogs
    ]
    return ecogs
