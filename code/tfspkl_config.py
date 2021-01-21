import json
import os


def return_config_dict():
    """Return configuration information

    Returns:
        dict: configuration information

    Misc:
        exclude_words_class: exclude words from the classifier vocabulary
        exclude_words: exclude words from the tranformer vocabulary
        log_interval:
    """
    CONFIG = {
        "begin_token":
        "<s>",
        "datum_suffix": ("conversation_trimmed", "trimmed"),
        "end_token":
        "</s>",
        "exclude_words_class": [
            "sp", "{lg}", "{ns}", "it", "a", "an", "and", "are", "as", "at",
            "be", "being", "by", "for", "from", "is", "of", "on", "that",
            "that's", "the", "there", "there's", "this", "to", "their", "them",
            "these", "he", "him", "his", "had", "have", "was", "were", "would"
        ],
        "exclude_words": ["sp", "{lg}", "{ns}"],
        "log_interval":
        32,
        "data_dir":
        os.path.join(os.getcwd(), 'data'),
        "num_cpus":
        8,
        "oov_token":
        "<unk>",
        "pad_token":
        "<pad>",
        "print_pad":
        120,
        "train_convs":
        '-train-convs.txt',
        "valid_convs":
        '-valid-convs.txt',
        "vocabulary":
        'std'
    }

    return CONFIG


def create_directory_paths(CONFIG, args):
    # Format directory logistics
    CONV_DIRS = [CONFIG["data_dir"] + '/%s/' % str(args.subject)]
    SAVE_DIR = os.path.join(os.getcwd(), 'results', str(args.subject))
    LOG_FILE = SAVE_DIR + 'output'
    PKL_DIR = os.path.join(SAVE_DIR, 'pickles')

    os.makedirs(PKL_DIR, exist_ok=True)

    DIR_DICT = dict(CONV_DIRS=CONV_DIRS,
                    SAVE_DIR=SAVE_DIR,
                    PKL_DIR=PKL_DIR,
                    LOG_FILE=LOG_FILE)
    CONFIG.update(DIR_DICT)

    return CONFIG


def build_config(args):
    """Combine configuration and input arguments

    Args:
        args (OrderedDict): parsed input arguments
        results_str (str): results folder name

    Returns:
        dict: combined configuration information
    """
    CONFIG = return_config_dict()

    if args.subject == 625:
        CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][0]]
    elif args.subject == 676:
        CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][1]]
    else:
        raise Exception('Wrong Subject ID')

    args.subject = str(args.subject)
    CONFIG.update(vars(args))

    if CONFIG["max_electrodes"]:
        CONFIG["electrode_list"] = [
            list(range(1, k + 1)) for k in CONFIG["max_electrodes"]
        ]
    else:
        CONFIG["max_electrodes"] = [
            len(item) for item in CONFIG["electrode_list"]
        ]

    CONFIG["num_features"] = sum(CONFIG["max_electrodes"])
    CONFIG = create_directory_paths(CONFIG, args)

    write_config(CONFIG)
    vars(args).update(CONFIG)

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


def read_config(results_folder):
    """Read configuration from to a file

    Args:
        results_folder (str): experiment folder from which to read config.json

    Returns:
        dict: configuration object
    """
    PROJ_FOLDER = os.getcwd()
    CONFIG_FILE_PATH = os.path.join(PROJ_FOLDER, results_folder, 'config.json')
    with open(CONFIG_FILE_PATH, 'r') as file_h:
        CONFIG = json.load(file_h)
    return CONFIG
