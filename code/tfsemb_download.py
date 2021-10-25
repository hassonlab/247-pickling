# Run python tfsemb_download

import os
from transformers import (BartForConditionalGeneration, BartTokenizer,
                          BertForMaskedLM, BertTokenizer, GPT2LMHeadModel,
                          GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer)
from utils import main_timer

def download_tokenizer_and_model(CACHE_DIR, tokenizer_class, model_class, model_name):
    model_class.from_pretrained(model_name,
                                output_hidden_states=True,
                                cache_dir=CACHE_DIR,
                                local_files_only=False)
    print('Downloaded model for', model_name)
    tokenizer_class.from_pretrained(model_name,
                                    add_prefix_space=True,
                                    cache_dir=CACHE_DIR,
                                    local_files_only=False)
    print('Downloaded tokenizer for', model_name)

def download_tokenizers_and_models():

    CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), '.cache')
    os.makedirs(CACHE_DIR, exist_ok=True)

    gpt2_models = ['gpt2','gpt2-large','gpt2-xl']

    for model in gpt2_models:
        download_tokenizer_and_model(CACHE_DIR, GPT2Tokenizer, GPT2LMHeadModel, model)

    download_tokenizer_and_model(CACHE_DIR, BertTokenizer, BertForMaskedLM, 'bert-large-uncased-whole-word-masking')

    # download_tokenizer_and_model(CACHE_DIR, RobertaTokenizer, RobertaForMaskedLM, 'roberta')
    # download_tokenizer_and_model(CACHE_DIR, BartTokenizer, BartForConditionalGeneration, 'bart')

    return

@main_timer
def main():
    download_tokenizers_and_models()

if __name__ == '__main__':
    main()