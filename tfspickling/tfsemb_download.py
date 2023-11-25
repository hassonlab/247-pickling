import os

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

CAUSAL_MODELS = [
    "gpt2",
    "gpt2-large",
    "gpt2-xl",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-30b",
    "bigscience/bloom",
]
SEQ2SEQ_MODELS = ["facebook/blenderbot_small-90M", "facebook/blenderbot-3B"]

MLM_MODELS = [
    # "gpt2-xl", # uncomment to run this model with MLM input
    # "gpt2-medium", # uncomment to run this model with MLM input
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "roberta-base",
    "roberta-large",
]

MODEL_CLASS_MAP = {
    "causal": (CAUSAL_MODELS, AutoModelForCausalLM),
    "seq2seq": (SEQ2SEQ_MODELS, AutoModelForSeq2SeqLM),
    "mlm": (MLM_MODELS, AutoModelForMaskedLM),
}


def clean_lm_model_name(item):
    """Remove unnecessary parts from the language model name.

    Args:
        item (str/list): full model name from HF Hub

    Returns:
        (str/list): pretty model name

    Example:
        clean_lm_model_name(EleutherAI/gpt-neo-1.3B) == 'gpt-neo-1.3B'
    """
    if isinstance(item, str):
        return item.split("/")[-1]

    if isinstance(item, list):
        return [clean_lm_model_name(i) for i in item]

    print("Invalid input. Please check.")


def get_max_context_length(model_name, tokenizer_class=None):
    """Return maximum possible context length for the model/tokenizer

    Args:
        model_name (str): Model name as seen on https://hugginface.co/models.
        tokenizer_class (Tokenizer, optional):
            Tokenizer class to be instantiated for the model.
            Defaults to None.

    Returns:
        int: Maximum allowed context for the model/tokenizer
    """
    tokenizer = download_hf_tokenizer(
        model_name,
        tokenizer_class=tokenizer_class,
        cache_dir=None,
        local_files_only=False,
    )

    return tokenizer.max_len_single_sentence


def get_model_num_layers(model_name):
    """Return number of hidden layers in the model

    Args:
        model_name (str): Model name as seen on https://hugginface.co/models.

    Returns:
        int: value of {n_layer|num_layers|num_hidden_layers} attribute
    """
    config = AutoConfig.from_pretrained(model_name)
    num_layers = getattr(
        config,
        "n_layer",
        getattr(
            config,
            "num_layers",
            getattr(config, "num_hidden_layers", None),
        ),
    )

    if not num_layers:
        print(f"{model_name} has no hidden layers")
        exit()

    return num_layers


def download_hf_model(
    model_name, model_class=None, cache_dir=None, local_files_only=False
):
    """Download a Huggingface model from the model repository (cache)."""
    if model_class is None:
        model_class = AutoModel

    if cache_dir is None:
        cache_dir = set_cache_dir()

    model = model_class.from_pretrained(
        model_name,
        output_hidden_states=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    return model


def download_hf_tokenizer(
    model_name, tokenizer_class=None, cache_dir=None, local_files_only=False
):
    """Download a Huggingface tokenizer from the model repository (cache)."""
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if cache_dir is None:
        cache_dir = set_cache_dir()

    tokenizer = tokenizer_class.from_pretrained(
        model_name,
        add_prefix_space=True,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    return tokenizer


def download_tokenizer_and_model(
    CACHE_DIR, tokenizer_class, model_class, model_name, local_files_only
):
    """Cache (or load) the model and tokenizer from the model repository (or cache).

    Args:
        CACHE_DIR (str): path where the model and tokenizer will be cached.
        tokenizer_class (Tokenizer): Tokenizer class to be instantiated for the model.
        model_class (Huggingface Model): Model class corresponding to model_name.
        model_name (str):  Model name as seen on https://hugginface.co/models.
        local_files_only (bool, optional): False (Default) if caching.
                                           True if loading from cache.

    Returns:
        tuple: (tokenizer, model)
    """
    print("Downloading model")
    model = download_hf_model(model_name, model_class, CACHE_DIR, local_files_only)

    print("Downloading tokenizer")
    tokenizer = download_hf_tokenizer(
        model_name, tokenizer_class, CACHE_DIR, local_files_only
    )

    return (model, tokenizer)


def set_cache_dir():
    CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), ".cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


def get_models_and_class(model_name):
    """Return the appropriate model class and model to download
    Args:
        model_name (str): Model name as seen on https://hugginface.co/models.

    Returns:
        models (list): model name or models in the same class
        mod_class (Huggingface Model): Model class corresponding to model_name.

    """
    models, mod_class = None, None
    for model_key, (model_list, model_class) in MODEL_CLASS_MAP.items():
        if model_name == model_key:
            models, mod_class = model_list, model_class
            break
        elif model_name in model_list:
            models, mod_class = [model_name], model_class
            break
        else:
            continue

    if not models or not model_class:
        print("Invalid Model List or Model Class")

    return models, mod_class


def download_tokenizers_and_models(model_name=None, local_files_only=False, debug=True):
    """This function downloads the tokenizer and model for the specified model name.

    Args:
        model_name (str, optional): Model name as seen on https://hugginface.co/models.
                                    Defaults to None.
        local_files_only (bool, optional): False (Default) if caching.
                                            True if loading from cache.
        debug (bool, optional): Check if caching was successful. Defaults to True.

    Returns:
        dict: Dictionary with model name as key and (tokenizer, model) as value.
    """
    CACHE_DIR = set_cache_dir()

    if model_name is None:
        print("Input argument cannot be empty")
        return

    models, model_class = get_models_and_class(model_name)

    model_dict = {}
    for model_name in models:
        print(f"Model Name: {model_name}")

        model_dict[model_name] = download_tokenizer_and_model(
            CACHE_DIR,
            AutoTokenizer,
            model_class,
            model_name,
            local_files_only,
        )

        # check if caching was successful
        if debug:
            print("Checking if model has been cached successfully")
            try:
                download_tokenizer_and_model(
                    CACHE_DIR,
                    AutoTokenizer,
                    model_class,
                    model_name,
                    True,
                )
            except:
                print(f"Caching of {model_name} failed")

    return model_dict


if __name__ == "__main__":
    download_tokenizers_and_models("causal", local_files_only=False, debug=True)
