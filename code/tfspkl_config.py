import json
import os


def create_directory_paths(args):
    # Format directory logistics
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    CONV_DIRS = DATA_DIR + '/%s/' % str(args.subject)
    SAVE_DIR = os.path.join(os.getcwd(), 'results', str(args.subject))
    PKL_DIR = os.path.join(SAVE_DIR, 'pickles')

    os.makedirs(PKL_DIR, exist_ok=True)

    DIR_DICT = dict(CONV_DIRS=CONV_DIRS,
                    SAVE_DIR=SAVE_DIR,
                    PKL_DIR=PKL_DIR)
    vars(args).update(DIR_DICT)

    return


def build_config(args):
    """Combine configuration and input arguments

    Args:
        args (OrderedDict): parsed input arguments
        results_str (str): results folder name

    Returns:
        dict: combined configuration information
    """
    args.exclude_words = ["sp", "{lg}", "{ns}"]
    create_directory_paths(args)

    write_config(vars(args))

    return args


def write_config(dictionary):
    """Write configuration to a file

    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join(dictionary['SAVE_DIR'], 'config.json')
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
