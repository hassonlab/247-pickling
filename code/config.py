import json
import os

import torch


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


def create_directory_paths(CONFIG, args, results_str):
    # Format directory logistics
    CONV_DIRS = [CONFIG["data_dir"] + '/%s/' % i for i in args.subjects]
    META_DIRS = [
        CONFIG["data_dir"] + '/%s-metadata/' % i for i in args.subjects
    ]
    if not args.output_folder:
        SAVE_DIR = './Results/%s-%s-%s-%s/' % (results_str, '+'.join(
            args.subjects), args.model, str(args.seed))
    else:
        SAVE_DIR = './%s/%s/%s/' % (args.exp_suffix, args.output_folder,
                                    str(args.seed))
    LOG_FILE = SAVE_DIR + 'output'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    DIR_DICT = dict(CONV_DIRS=CONV_DIRS,
                    META_DIRS=META_DIRS,
                    SAVE_DIR=SAVE_DIR,
                    LOG_FILE=LOG_FILE)
    CONFIG.update(DIR_DICT)

    return CONFIG


def build_config(args, results_str):
    """Combine configuration and input arguments

    Args:
        args (OrderedDict): parsed input arguments
        results_str (str): results folder name

    Returns:
        dict: combined configuration information
    """
    CONFIG = return_config_dict()
    gpus = min(args.gpus, torch.cuda.device_count())

    # Model objectives
    MODEL_OBJ = {
        "ConvNet10": "classifier",
        "PITOM": "classifier",
        "MeNTALmini": "classifier",
        "MeNTAL": "seq2seq"
    }

    args.model = args.model.split("_")[0]
    classify = False if (args.model in MODEL_OBJ
                         and MODEL_OBJ[args.model] == "seq2seq") else True

    if len(args.subjects) == 1:
        if args.subjects[0] == '625':
            CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][0]]
        elif args.subjects[0] == '676':
            CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][1]]

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

    DIR_DICT = dict(classify=classify, gpus=gpus)

    CONFIG = create_directory_paths(CONFIG, args, results_str)

    CONFIG.update(DIR_DICT)
    write_config(CONFIG)

    return CONFIG


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
