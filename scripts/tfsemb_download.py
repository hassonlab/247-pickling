# Run python tfsemb_download

import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

CAUSAL_MODELS = [
    "gpt2",
    "gpt2-xl",
    "gpt2-large",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-1.3B",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-30b",
]
SEQ2SEQ_MODELS = ["facebook/blenderbot_small-90M"]

# TODO: Add MLM_MODELS (Masked Language Models  )


def download_tokenizer_and_model(CACHE_DIR, tokenizer_class, model_class, model_name):
    model_class.from_pretrained(
        model_name,
        output_hidden_states=True,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )
    print("Downloaded model for", model_name)
    tokenizer_class.from_pretrained(
        model_name,
        add_prefix_space=True,
        cache_dir=CACHE_DIR,
        local_files_only=False,
    )
    print("Downloaded tokenizer for", model_name)


def download_tokenizers_and_models(model_name=None):

    CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), ".cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if model_name is None:
        print("Input argument cannot be empty")
        exit(1)

    if model_name == "causal" or model_name in CAUSAL_MODELS:
        model_class = AutoModelForCausalLM
        MODELS = CAUSAL_MODELS if model_name == "causal" else [model_name]
    elif model_name == "seq2seq":
        model_class = AutoModelForSeq2SeqLM
        MODELS = SEQ2SEQ_MODELS if model_name == "seq2seq" else [model_name]
    else:
        print("Invalid Model Name")
        exit(1)

    for model in MODELS:
        download_tokenizer_and_model(
            CACHE_DIR,
            AutoTokenizer,
            model_class,
            model,
        )

    return


if __name__ == "__main__":
    download_tokenizers_and_models("causal")
