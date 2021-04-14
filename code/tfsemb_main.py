import argparse
import os
import pickle
import string
from itertools import islice

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from transformers import (BartForConditionalGeneration, BartTokenizer,
                          BertForMaskedLM, BertTokenizer, GPT2LMHeadModel,
                          GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer)
from utils import lcs, main_timer


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = file_name + add_ext
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def select_conversation(args, df):
    if args.conversation_id:
        df = df[df.conversation_id == args.conversation_id]
    return df


def load_pickle(args):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(args.pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum['labels'])

    return df


def add_glove_embeddings(df, dim=None):
    if dim == 50:
        glove = api.load('glove-wiki-gigaword-50')
        df['glove50_embeddings'] = df['token2word'].apply(
            lambda x: get_vector(x, glove))
    else:
        raise Exception("Incorrect glove dimension")

    return df


def check_token_is_root(args, df):
    if args.embedding_type == 'gpt2-xl':
        df['gpt2-xl_token_is_root'] = df['word'] == df['token'].apply(
            args.tokenizer.convert_tokens_to_string).str.strip()
    elif args.embedding_type == 'bert':
        df['bert_token_is_root'] = df['word'] == df['token']
    else:
        raise Exception("embedding type doesn't exist")

    return df


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def convert_token_to_idx(df, tokenizer):
    df['token_id'] = df['token'].apply(tokenizer.convert_tokens_to_ids)
    return df


def tokenize_and_explode(args, df):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df['token'] = df.word.apply(args.tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)
    df['token2word'] = df['token'].apply(
        args.tokenizer.convert_tokens_to_string).str.strip().str.lower()
    df = convert_token_to_idx(df, args.tokenizer)
    df = check_token_is_root(args, df)
    df = add_glove_embeddings(df, dim=50)

    return df


def get_token_indices(args, num_tokens):
    if args.embedding_type == 'gpt2-xl':
        start, stop = 0, num_tokens
    elif args.embedding_type == 'bert':
        start, stop = 1, num_tokens + 1
    else:
        raise Exception('wrong model')

    return (start, stop)


def map_embeddings_to_tokens(args, df, embed):

    multi = df.set_index(['conversation_id', 'sentence_idx', 'sentence'])
    unique_sentence_idx = multi.index.unique().values

    uniq_sentence_count = len(get_unique_sentences(df))
    assert uniq_sentence_count == len(embed)

    c = []
    for unique_idx, sentence_embedding in zip(unique_sentence_idx, embed):
        a = df['conversation_id'] == unique_idx[0]
        b = df['sentence_idx'] == unique_idx[1]
        num_tokens = sum(a & b)
        start, stop = get_token_indices(args, num_tokens)
        c.append(pd.Series(sentence_embedding[start:stop, :].tolist()))

    df['embeddings'] = pd.concat(c, ignore_index=True)
    return df


def get_unique_sentences(df):
    return df[['conversation_id', 'sentence_idx',
               'sentence']].drop_duplicates()['sentence'].tolist()


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) <= n:
        yield result
    for elem in it:
        result = result[1:] + (elem, )
        yield result


def process_extracted_embeddings(concat_output):
    """(batch_size, max_len, embedding_size)"""
    # concatenate all batches
    concatenated_embeddings = torch.cat(concat_output, dim=0).numpy()
    emb_dim = concatenated_embeddings.shape[-1]

    # the first token is always empty
    init_token_embedding = np.empty((1, emb_dim)) * np.nan

    extracted_embeddings = np.concatenate(
        [init_token_embedding, concatenated_embeddings], axis=0)

    return extracted_embeddings


def process_extracted_logits(args, concat_logits, sentence_token_ids):
    """Get the probability for the _correct_ word"""
    # (batch_size, max_len, vocab_size)

    # concatenate all batches
    prediction_scores = torch.cat(concat_logits, axis=0)

    if prediction_scores.shape[0] == 0:
        return [None], [None], [None]
    elif prediction_scores.shape[0] == 1:
        true_y = torch.tensor(sentence_token_ids[0][1:]).unsqueeze(-1)
    else:
        sti = torch.tensor(sentence_token_ids)
        true_y = torch.cat([sti[0, 1:], sti[1:, -1]]).unsqueeze(-1)

    prediction_probabilities = F.softmax(prediction_scores, dim=1)

    logp = np.log2(prediction_probabilities)
    entropy = [None] + torch.sum(-prediction_probabilities * logp,
                                 dim=1).tolist()

    top1_probabilities, top1_probabilities_idx = prediction_probabilities.max(
        dim=1)
    predicted_tokens = args.tokenizer.convert_ids_to_tokens(
        top1_probabilities_idx)
    predicted_words = [
        args.tokenizer.convert_tokens_to_string(token)
        for token in predicted_tokens
    ]

    # top-1 probabilities
    top1_probabilities = [None] + top1_probabilities.tolist()
    # top-1 word
    top1_words = [None] + predicted_words
    # probability of correct word
    true_y_probability = [None] + prediction_probabilities.gather(
        1, true_y).squeeze(-1).tolist()
    #TODO: probabilities of all words

    return top1_words, top1_probabilities, true_y_probability, entropy


def extract_select_vectors(batch_idx, array):
    if batch_idx == 0:
        x = array[0, :-1, :].clone()
        if array.shape[0] > 1:
            rem_sentences_preds = array[1:, -2, :].clone()
            x = torch.cat([x, rem_sentences_preds], axis=0)
    else:
        x = array[:, -2, :].clone()

    return x


def model_forward_pass(args, data_dl):
    model = args.model
    device = args.device

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        all_embeddings = []
        all_logits = []
        for batch_idx, batch in enumerate(data_dl):
            batch = batch.to(args.device)
            model_output = model(batch)

            embeddings = model_output.hidden_states[-1].cpu()
            logits = model_output.logits.cpu()

            embeddings = extract_select_vectors(batch_idx, embeddings)
            logits = extract_select_vectors(batch_idx, logits)

            all_embeddings.append(embeddings)
            all_logits.append(logits)

    return all_embeddings, all_logits


def get_conversation_tokens(df, conversation):
    token_list = df[df.conversation_id == conversation]['token_id'].tolist()
    return token_list


def make_input_from_tokens(args, token_list):
    windows = list(window(token_list, args.context_length))
    return windows


def make_dataloader_from_input(windows):
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=2, shuffle=False)
    return data_dl


def generate_embeddings_with_context(args, df):
    df = tokenize_and_explode(args, df)
    if args.embedding_type == 'gpt2-xl':
        args.tokenizer.pad_token = args.tokenizer.eos_token

    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    for conversation in df.conversation_id.unique():
        token_list = get_conversation_tokens(df, conversation)
        model_input = make_input_from_tokens(args, token_list)
        input_dl = make_dataloader_from_input(model_input)
        embeddings, logits = model_forward_pass(args, input_dl)

        embeddings = process_extracted_embeddings(embeddings)
        assert embeddings.shape[0] == len(token_list)
        final_embeddings.append(embeddings)

        top1_word, top1_prob, true_y_prob, entropy = process_extracted_logits(
            args, logits, model_input)
        final_top1_word.extend(top1_word)
        final_top1_prob.extend(top1_prob)
        final_true_y_prob.extend(true_y_prob)

    # TODO: convert embeddings dtype from object to float
    df['embeddings'] = np.concatenate(final_embeddings, axis=0).tolist()
    df['top1_pred'] = final_top1_word
    df['top1_pred_prob'] = final_top1_prob
    df['true_pred_prob'] = final_true_y_prob
    df['surprise'] = -df['true_pred_prob'] * np.log2(df['true_pred_prob'])
    df['entropy'] = entropy

    return df


def generate_embeddings(args, df):
    tokenizer = args.tokenizer
    model = args.model
    device = args.device

    model = model.to(device)
    model.eval()

    df = tokenize_and_explode(args, df)
    unique_sentence_list = get_unique_sentences(df)

    if args.embedding_type == 'gpt2-xl':
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(unique_sentence_list, padding=True, return_tensors='pt')
    input_ids_val = tokens['input_ids']
    attention_masks_val = tokens['attention_mask']

    dataset = data.TensorDataset(input_ids_val, attention_masks_val)
    data_dl = data.DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        concat_output = []
        for batch in data_dl:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            model_output = model(**inputs)
            concat_output.append(model_output[-1][-1].detach().cpu().numpy())

    embeddings = np.concatenate(concat_output, axis=0)
    emb_df = map_embeddings_to_tokens(args, df, embeddings)

    return emb_df


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def generate_glove_embeddings(args, df):
    glove = api.load('glove-wiki-gigaword-50')
    df['embeddings'] = df['word'].apply(lambda x: get_vector(x, glove))

    return df


def setup_environ(args):

    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), 'results', args.project_id)
    PKL_DIR = os.path.join(RESULTS_DIR, args.subject, 'pickles')

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

    labels_file = '_'.join([args.subject, args.pkl_identifier, 'labels.pkl'])
    args.pickle_name = os.path.join(PKL_DIR, labels_file)

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(os.listdir(args.input_dir))

    args.gpus = torch.cuda.device_count()
    if args.gpus > 1:
        args.model = nn.DataParallel(args.model)

    stra = '_'.join([args.embedding_type, 'cnxt', str(args.context_length)])

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(RESULTS_DIR, args.subject, 'embeddings',
                                       stra, args.pkl_identifier)
        os.makedirs(args.output_dir, exist_ok=True)

        output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir, output_file_name)

        args.output_file_prefinal = os.path.join(
            args.output_dir, output_file_name + '_prefinal')

    return


def select_tokenizer_and_model(args):

    if args.embedding_type == 'gpt2-xl':
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
        model_name = 'gpt2-xl'
    elif args.embedding_type == 'roberta':
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMaskedLM
        model_name = 'roberta'
    elif args.embedding_type == 'bert':
        tokenizer_class = BertTokenizer
        model_class = BertForMaskedLM
        model_name = 'bert-large-uncased-whole-word-masking'
    elif args.embedding_type == 'bart':
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration
        model_name = 'bart'
    elif args.embedding_type == 'glove50':
        return
    else:
        print('No model found for', args.model_name)
        exit(1)

    CACHE_DIR = os.path.join(os.path.dirname(os.getcwd()), '.cache')
    os.makedirs(CACHE_DIR, exist_ok=True)

    args.model = model_class.from_pretrained(model_name,
                                             output_hidden_states=True,
                                             cache_dir=CACHE_DIR)
    args.tokenizer = tokenizer_class.from_pretrained(model_name,
                                                     add_prefix_space=True,
                                                     cache_dir=CACHE_DIR)

    if args.history and args.context_length <= 0:
        args.context_length = args.tokenizer.max_len_single_sentence
        assert args.context_length <= args.tokenizer.max_len_single_sentence, \
            'given length is greater than max length'

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        default='bert-large-uncased-whole-word-masking')
    parser.add_argument('--embedding-type', type=str, default='glove')
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-hidden-states',
                        action='store_true',
                        default=False)
    parser.add_argument('--subject', type=str, default='625')
    parser.add_argument('--history', action='store_true', default=False)
    parser.add_argument('--conversation-id', type=int, default=0)
    parser.add_argument('--pkl-identifier', type=str, default=None)
    parser.add_argument('--project-id', type=str, default=None)

    return parser.parse_args()


def return_story_as_df(args):
    """Tokenize the podcast transcript and return as dataframe

    Args:
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
    story_file = os.path.join(DATA_DIR, 'podcast-transcription.txt')

    # Read all words and tokenize them
    with open(story_file, 'r') as fp:
        data = fp.readlines()

        data = [item.split(' ') for item in data]
        data = [
            item[:-2] + [' '.join(item[-2:])] if item[-1] == '\n' else item
            for item in data
        ]
        data = [item for sublist in data for item in sublist]

    df = pd.DataFrame(data, columns=['word'])[:100]
    df['conversation_id'] = 1

    return df


@main_timer
def main():
    args = parse_arguments()
    select_tokenizer_and_model(args)
    setup_environ(args)

    if args.project_id == 'tfs':
        utterance_df = load_pickle(args)
        utterance_df = select_conversation(args, utterance_df)
    elif args.project_id == 'podcast':
        utterance_df = return_story_as_df(args)
    else:
        raise Exception('Invalid Project ID')

    if args.history:
        if args.embedding_type == 'gpt2-xl':
            df = generate_embeddings_with_context(args, utterance_df)
        else:
            print('TODO: Generate embeddings for this model with context')
    else:
        if args.embedding_type == 'glove50':
            df = generate_glove_embeddings(args, utterance_df)
        else:
            df = generate_embeddings(args, utterance_df)

    if args.project_id == 'podcast':
        DATA_DIR = os.path.join(os.getcwd(), 'data', args.project_id)
        cloze_file = os.path.join(DATA_DIR, 'podcast-datum-cloze.csv')

        # Align the two lists
        cloze_df = pd.read_csv(cloze_file, sep=',')[:100]
        words = list(map(str.lower, cloze_df.word.tolist()))

        model_tokens = df['token2word'].tolist()

        mask1, mask2 = lcs(words, model_tokens)

        cloze_df = cloze_df.iloc[mask1, :].reset_index(drop=True)
        df = df.iloc[mask2, :].reset_index(drop=True)

        df_final = pd.concat([df, cloze_df], axis=1)
        df = df_final.loc[:, ~df_final.columns.duplicated()]

    save_pickle(df.to_dict('records'), args.output_file)

    return


if __name__ == '__main__':
    main()
