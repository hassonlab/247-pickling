import json
import os


def create_directory_paths(args):
    # Format directory logistics
    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    CONV_DIRS = DATA_DIR + '/%s/' % args.subject
    SAVE_DIR = os.path.join(os.getcwd(), 'results', args.project_id,
                            args.subject)
    PKL_DIR = os.path.join(SAVE_DIR, 'pickles')

    os.makedirs(PKL_DIR, exist_ok=True)

    DIR_DICT = dict(CONV_DIRS=CONV_DIRS,
                    SAVE_DIR=SAVE_DIR,
                    PKL_DIR=PKL_DIR,
                    DATA_DIR=DATA_DIR)
    vars(args).update(DIR_DICT)

    if args.sig_elec_file:
        sig_file_path, sig_file_name = os.path.split(args.sig_elec_file)
        if not sig_file_path:
            args.sig_elec_file = os.path.join(DATA_DIR, sig_file_name)

    return


def build_config(args):
    """Combine configuration and input arguments

    Args:
        args (OrderedDict): parsed input arguments
        results_str (str): results folder name

    Returns:
        dict: combined configuration information
    """
    args.subject = str(args.subject)
    args.exclude_words = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP"]
    args.non_words = ['hm', 'huh', 'mhm', 'mm', 'oh', 'uh', 'uhuh', 'um']
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
