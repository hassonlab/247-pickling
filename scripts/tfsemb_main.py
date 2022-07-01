import argparse
import os
import pickle
import string

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import tfsemb_download as tfsemb_dwnld
from utils import main_timer


def save_pickle(args, item, file_name, embeddings=None):
    """Write 'item' to 'file_name.pkl'"""
    add_ext = "" if file_name.endswith(".pkl") else ".pkl"

    file_name = file_name + add_ext

    if embeddings is not None:
        for layer_idx, embedding in embeddings.items():
            item["embeddings"] = embedding.tolist()
            filename = file_name % layer_idx
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as fh:
                pickle.dump(item.to_dict("records"), fh)
    else:
        filename = file_name % args.layer_idx[0]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as fh:
            pickle.dump(item.to_dict("records"), fh)
    return


def select_conversation(args, df):
    if args.conversation_id:
        print("Selecting conversation", args.conversation_id)
        df = df[df.conversation_id == args.conversation_id]
    return df


def load_pickle(args):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(args.pickle_name, "rb") as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum["labels"])

    return df


def add_glove_embeddings(df, dim=None):
    if dim == 50:
        glove = api.load("glove-wiki-gigaword-50")
        df["glove50_embeddings"] = df["token2word"].apply(
            lambda x: get_vector(x, glove)
        )
    else:
        raise Exception("Incorrect glove dimension")

    return df


def check_token_is_root(args, df):
    token_is_root_string = args.embedding_type.split("/")[-1] + "_token_is_root"
    df[token_is_root_string] = (
        df["word"]
        == df["token"].apply(args.tokenizer.convert_tokens_to_string).str.strip()
    )

    return df


def remove_punctuation(df):
    return df[~df.token.isin(list(string.punctuation))]


def convert_token_to_idx(df, tokenizer):
    df["token_id"] = df["token"].apply(tokenizer.convert_tokens_to_ids)
    return df


def tokenize_and_explode(args, df):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df["token"] = df.word.apply(args.tokenizer.tokenize)
    df = df.explode("token", ignore_index=True)
    df["token2word"] = (
        df["token"]
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
        .str.lower()
    )
    df = convert_token_to_idx(df, args.tokenizer)
    df = check_token_is_root(args, df)

    # Add a token index for each word's token
    for value in df["index"].unique():
        if value is not None:
            flag = df["index"] == value
            df.loc[flag, "token_idx"] = np.arange(sum(flag))

    return df


def get_token_indices(args, num_tokens):
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        start, stop = 0, num_tokens
    # elif args.embedding_type == "bert":
    #     start, stop = 1, num_tokens + 1
    else:
        raise Exception("wrong model")

    return (start, stop)


def map_embeddings_to_tokens(args, df, embed):

    multi = df.set_index(["conversation_id", "sentence_idx", "sentence"])
    unique_sentence_idx = multi.index.unique().values

    uniq_sentence_count = len(get_unique_sentences(df))
    assert uniq_sentence_count == len(embed)

    c = []
    for unique_idx, sentence_embedding in zip(unique_sentence_idx, embed):
        a = df["conversation_id"] == unique_idx[0]
        b = df["sentence_idx"] == unique_idx[1]
        num_tokens = sum(a & b)
        start, stop = get_token_indices(args, num_tokens)
        c.append(pd.Series(sentence_embedding[start:stop, :].tolist()))

    df["embeddings"] = pd.concat(c, ignore_index=True)
    return df


def get_unique_sentences(df):
    return (
        df[["conversation_id", "sentence_idx", "sentence"]]
        .drop_duplicates()["sentence"]
        .tolist()
    )


def process_extracted_embeddings(args, concat_output):
    """(batch_size, max_len, embedding_size)"""
    # concatenate all batches
    concatenated_embeddings = torch.cat(concat_output, dim=0).numpy()
    extracted_embeddings = concatenated_embeddings

    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        emb_dim = concatenated_embeddings.shape[-1]

        # the first token is always empty
        init_token_embedding = np.empty((1, emb_dim)) * np.nan

        extracted_embeddings = np.concatenate(
            [init_token_embedding, concatenated_embeddings], axis=0
        )

    return extracted_embeddings


def process_extracted_embeddings_all_layers(args, layer_embeddings_dict):
    layer_embeddings = dict()
    for layer_idx in args.layer_idx:
        concat_output = []
        for item_dict in layer_embeddings_dict:
            concat_output.append(item_dict[layer_idx])
        layer_embeddings[layer_idx] = process_extracted_embeddings(args, concat_output)

    return layer_embeddings


def process_extracted_logits(args, concat_logits, sentence_token_ids):
    """Get the probability for the _correct_ word"""
    # (batch_size, max_len, vocab_size)

    # concatenate all batches
    prediction_scores = torch.cat(concat_logits, axis=0)
    if "blenderbot" in args.embedding_type:
        true_y = torch.tensor(sentence_token_ids).unsqueeze(-1)
    else:
        if prediction_scores.shape[0] == 0:
            return [None], [None], [None]
        elif prediction_scores.shape[0] == 1:
            true_y = torch.tensor(sentence_token_ids[0][1:]).unsqueeze(-1)
        else:
            sti = torch.tensor(sentence_token_ids)
            true_y = torch.cat([sti[0, 1:], sti[1:, -1]]).unsqueeze(-1)

    prediction_probabilities = F.softmax(prediction_scores, dim=1)

    logp = np.log2(prediction_probabilities)
    entropy = [None] + torch.sum(-prediction_probabilities * logp, dim=1).tolist()

    top1_probabilities, top1_probabilities_idx = prediction_probabilities.max(dim=1)
    predicted_tokens = args.tokenizer.convert_ids_to_tokens(top1_probabilities_idx)
    predicted_words = predicted_tokens
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        predicted_words = [
            args.tokenizer.convert_tokens_to_string(token) for token in predicted_tokens
        ]

    # top-1 probabilities
    top1_probabilities = [None] + top1_probabilities.tolist()
    # top-1 word
    top1_words = [None] + predicted_words
    # probability of correct word
    true_y_probability = [None] + prediction_probabilities.gather(1, true_y).squeeze(
        -1
    ).tolist()
    # TODO: probabilities of all words

    return top1_words, top1_probabilities, true_y_probability, entropy


def extract_select_vectors(batch_idx, array):
    if batch_idx == 0:
        x = array[0, :-1, :].clone()
        if array.shape[0] > 1:
            try:
                rem_sentences_preds = array[1:, -2, :].clone()
            except:
                rem_sentences_preds = array[1:, -1, :].clone()

            x = torch.cat([x, rem_sentences_preds], axis=0)
    else:
        try:
            x = array[:, -2, :].clone()
        except:
            x = array[:, -1, :].clone()

    return x


def extract_select_vectors_all_layers(batch_idx, array, layers=None):

    array_actual = tuple(y.cpu() for y in array)

    all_layers_x = dict()
    for layer_idx in layers:
        array = array_actual[layer_idx]
        all_layers_x[layer_idx] = extract_select_vectors(batch_idx, array)

    return all_layers_x


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

            logits = model_output.logits.cpu()

            embeddings = extract_select_vectors_all_layers(
                batch_idx, model_output.hidden_states, args.layer_idx
            )
            logits = extract_select_vectors(batch_idx, logits)

            all_embeddings.append(embeddings)
            all_logits.append(logits)

    return all_embeddings, all_logits


def transformer_forward_pass(args, data_dl):
    """Forward pass through full transformer encoder and decoder."""
    model = args.model
    device = args.device

    # Example conversation:
    #                                           | <s> good morning
    # <s> good morning </s>                     | <s> how are you
    # <s> good morning </s> <s> how are you </s>| <s> i'm good and you

    encoderlayers = np.arange(1, 9)
    decoderlayers = encoderlayers + 8
    encoderkey = "encoder_hidden_states"
    decoderkey = "decoder_hidden_states"
    accuracy, count = 0, 0

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        all_embeddings = []
        all_logits = []
        for batch_idx, batch in enumerate(data_dl):
            input_ids = torch.LongTensor(batch["encoder_ids"]).to(device)
            decoder_ids = torch.LongTensor(batch["decoder_ids"]).to(device)
            outputs = model(
                input_ids.unsqueeze(0), decoder_input_ids=decoder_ids.unsqueeze(0)
            )
            # After: get all relevant layers
            embeddings = {
                i: outputs[decoderkey][i - 8].cpu()[0, :-1, :] for i in decoderlayers
            }
            logits = outputs.logits.cpu()[0, :-1, :]

            if batch_idx > 0:
                prev_ntokens = len(all_embeddings[-1][9]) + 1  # previous tokens
                for token_idx in range(prev_ntokens - 1):
                    if token_idx == 0:
                        portion = (0, slice(-prev_ntokens, -1), slice(512))
                        encoder_embs = {
                            i: outputs[encoderkey][i][portion].cpu()
                            for i in encoderlayers
                        }  # take embeddings with original model (all word tokens)
                    else:
                        input_ids = torch.cat(
                            [input_ids[0:-2], input_ids[-1:]]
                        )  # delete last word token
                        outputs = model(
                            input_ids.unsqueeze(0),
                            decoder_input_ids=decoder_ids.unsqueeze(0),
                        )  # rerun model
                        portion = (
                            0,
                            slice(-2, -1),
                            slice(512),
                        )  # second to last token embedding
                        for i in encoderlayers:
                            encoder_embs[i][-token_idx - 1] = outputs[encoderkey][i][
                                portion
                            ].cpu()  # update embeddings
                all_embeddings[-1].update(encoder_embs)
                # [all_embeddings[-1][i].shape for i in range(1, 17)]
                # tokenizer = args.tokenizer
                # print(tokenizer.convert_ids_to_tokens(data_dl[batch_idx]['decoder_ids']))
                if batch_idx == len(data_dl) - 1:
                    continue

            all_embeddings.append(embeddings)
            all_logits.append(logits)

            # Just to compute accuracy
            predictions = outputs.logits.cpu().numpy().argmax(axis=-1)
            y_true = decoder_ids[1:].cpu().numpy()
            y_pred = predictions[0, :-1]
            accuracy += np.sum(y_true == y_pred)
            count += y_pred.size

            # # Uncomment to debug
            # tokenizer = args.tokenizer
            # print(tokenizer.decode(batch['encoder_ids']))
            # print(tokenizer.decode(batch['decoder_ids']))
            # print(tokenizer.convert_ids_to_tokens(batch['decoder_ids'][1:]))
            # print(tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1).squeeze().tolist()))
            # print()
            # breakpoint()

    # assert len(all_embeddings) == len(data_dl) - 1
    # assert sum([len(e[1]) for e in all_embeddings]) == sum([len(d['decoder_ids'])-1 for d in data_dl])
    print("model_forward accuracy", accuracy / count)
    return all_embeddings, all_logits


def get_conversation_tokens(df, conversation):
    token_list = df[df.conversation_id == conversation]["token_id"].tolist()
    return token_list


def make_conversational_input(args, df):
    """
    Create a conversational context/response pair to be fed into an encoder
    decoder transformer architecture. The context is a series of utterances
    that precede a new utterance response.

    examples = [
        {
            'encoder_inputs': [<s> </s>]
            'decoder_inputs': [<s> hi, how are you]
        },
        {
            'encoder_inputs': [<s> hi, how are you </s> <s> ok good </s>]
            'decoder_inputs': [<s> i'm doing fine]
        },
        {
            'encoder_inputs': [<s> hi, how are you </s> <s> ok good </s> <s> i'm doing fine </s> ]
            'decoder_inputs': [<s> ...]
        },
    ]
    """

    bos = args.tokenizer.bos_token_id
    eos = args.tokenizer.eos_token_id
    sep = args.tokenizer.sep_token_id

    sep_id = [sep] if sep is not None else [eos]
    bos_id = [bos] if bos is not None else [sep]
    convo = [
        bos_id + row.token_id.values.tolist() + sep_id
        for _, row in df.groupby("sentence_idx")
    ]

    # add empty context at begnning to get states of first utterance
    # add empty context at the end to get encoder states of last utterance
    convo = [[eos]] + convo + [[eos, eos]]

    def create_context(conv, last_position, max_tokens=128):
        if last_position == 0:
            return conv[0]
        ctx = []
        for p in range(last_position, 0, -1):
            if len(ctx) + len(conv[p]) > max_tokens:
                break
            ctx = conv[p] + ctx
        return ctx

    examples = []
    for j, response in enumerate(convo):
        if j == 0:
            continue
        context = create_context(convo, j - 1)
        if len(context) > 0:
            examples.append({"encoder_ids": context, "decoder_ids": response[:-1]})

    # Ensure we maintained correct number of tokens per utterance
    first = np.array([len(e["decoder_ids"]) - 1 for e in examples])
    second = df.sentence_idx.value_counts(sort=False).sort_index()
    # minus 1 because we add an extra utterance for encoder
    assert len(examples) - 1 == len(second), "number of utts doesn't match"
    assert (first[:-1] == second).all(), "number of tokens per utt is bad"
    # (second.values != first).nonzero()[0][0]
    # len(input_dl[-4]['decoder_ids'])-1
    # print(args.tokenizer.decode(input_dl[578]['decoder_ids']))
    # df_convo[df_convo.sentence_idx == 600]

    return examples


def printe(example, args):
    tokenizer = args.tokenizer
    print(tokenizer.decode(example["encoder_ids"]))
    print(tokenizer.convert_ids_to_tokens(example["decoder_ids"]))
    print()


def generate_conversational_embeddings(args, df):
    df = tokenize_and_explode(args, df)
    # This is a workaround. Blenderbot is limited to 128 tokens so having
    # long utterances breaks that. We remove them here, as well as the next
    # utterance to keep the turn taking the same.
    utt_lens = df.sentence_idx.value_counts(sort=False)
    long_utts = utt_lens.index[utt_lens > 128 - 2].values
    long_utts = np.concatenate((long_utts, long_utts + 1))
    df = df[~df.sentence_idx.isin(long_utts)]
    print("Removing long utterances", long_utts)
    assert len(df), "No utterances left after"

    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    for conversation in df.conversation_id.unique():
        df_convo = df[df.conversation_id == conversation]

        # Create input and push through model
        input_dl = make_conversational_input(args, df_convo)
        embeddings, logits = transformer_forward_pass(args, input_dl)

        embeddings = process_extracted_embeddings_all_layers(args, embeddings)
        for _, item in embeddings.items():
            assert item.shape[0] == len(df_convo)
        final_embeddings.append(embeddings)

        y_true = np.concatenate([e["decoder_ids"][1:] for e in input_dl[:-1]])
        top1_word, top1_prob, true_y_prob, entropy = process_extracted_logits(
            args, logits, y_true
        )

        # Remove first None that is added by the previous function
        final_top1_word.extend(top1_word[1:])
        final_top1_prob.extend(top1_prob[1:])
        final_true_y_prob.extend(true_y_prob[1:])

    df["top1_pred"] = final_top1_word
    df["top1_pred_prob"] = final_top1_prob
    df["true_pred_prob"] = final_true_y_prob
    df["surprise"] = -df["true_pred_prob"] * np.log2(df["true_pred_prob"])
    print("Accuracy", (df.token == df.top1_pred).mean())

    if len(final_embeddings) > 1:
        # TODO concat all embeddings and return a dictionary
        # previous: np.concatenate(final_embeddings, axis=0)
        raise NotImplementedError
    else:
        final_embeddings = final_embeddings[0]

    return df, final_embeddings


def make_input_from_tokens(args, token_list):
    size = args.context_length

    if len(token_list) <= size:
        windows = [tuple(token_list)]
    else:
        windows = [
            tuple(token_list[x : x + size]) for x in range(len(token_list) - size + 1)
        ]

    return windows


def make_dataloader_from_input(windows):
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=1, shuffle=False)
    return data_dl


def generate_causal_embeddings(args, df):
    df = tokenize_and_explode(args, df)
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
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

        embeddings = process_extracted_embeddings_all_layers(args, embeddings)
        for _, item in embeddings.items():
            assert item.shape[0] == len(token_list)
        final_embeddings.append(embeddings)

        top1_word, top1_prob, true_y_prob, entropy = process_extracted_logits(
            args, logits, model_input
        )
        final_top1_word.extend(top1_word)
        final_top1_prob.extend(top1_prob)
        final_true_y_prob.extend(true_y_prob)

    if len(final_embeddings) > 1:
        # TODO concat all embeddings and return a dictionary
        # previous: np.concatenate(final_embeddings, axis=0)
        raise NotImplementedError
    else:
        final_embeddings = final_embeddings[0]

    df["top1_pred"] = final_top1_word
    df["top1_pred_prob"] = final_top1_prob
    df["true_pred_prob"] = final_true_y_prob
    df["surprise"] = -df["true_pred_prob"] * np.log2(df["true_pred_prob"])
    df["entropy"] = entropy

    return df, final_embeddings


def generate_embeddings(args, df):
    tokenizer = args.tokenizer
    model = args.model
    device = args.device

    model = model.to(device)
    model.eval()
    df = tokenize_and_explode(args, df)
    unique_sentence_list = get_unique_sentences(df)

    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(unique_sentence_list, padding=True, return_tensors="pt")
    input_ids_val = tokens["input_ids"]
    attention_masks_val = tokens["attention_mask"]
    dataset = data.TensorDataset(input_ids_val, attention_masks_val)
    data_dl = data.DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        concat_output = []
        for batch in data_dl:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
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
    glove = api.load("glove-wiki-gigaword-50")
    df["embeddings"] = df["word"].apply(lambda x: get_vector(x.lower(), glove))

    return df


def setup_environ(args):

    DATA_DIR = os.path.join(os.getcwd(), "data", args.project_id)
    RESULTS_DIR = os.path.join(os.getcwd(), "results", args.project_id)
    PKL_DIR = os.path.join(RESULTS_DIR, args.subject, "pickles")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels_file = "_".join([args.subject, args.pkl_identifier, "labels.pkl"])
    args.pickle_name = os.path.join(PKL_DIR, labels_file)

    args.input_dir = os.path.join(DATA_DIR, args.subject)
    args.conversation_list = sorted(os.listdir(args.input_dir))

    args.gpus = torch.cuda.device_count()
    if args.gpus > 1:
        args.model = nn.DataParallel(args.model)

    stra = f'{args.embedding_type.split("/")[-1]}_cnxt_{args.context_length}'

    # TODO: if multiple conversations are specified in input
    if args.conversation_id:
        args.output_dir = os.path.join(
            RESULTS_DIR,
            args.subject,
            "embeddings",
            stra,
            args.pkl_identifier,
            "layer_%02d",
        )
        output_file_name = args.conversation_list[args.conversation_id - 1]
        args.output_file = os.path.join(args.output_dir, output_file_name)

    return


def get_model_layer_count(args):
    model = args.model
    max_layers = getattr(
        model.config,
        "n_layer",
        getattr(
            model.config, "num_layers", getattr(model.config, "num_hidden_layers", None)
        ),
    )

    # NOTE: layer_idx is shifted by 1 because the first item in hidden_states
    # corresponds to the output of the embeddings_layer
    if args.layer_idx == "all":
        args.layer_idx = np.arange(1, max_layers + 1)
    elif args.layer_idx == "last":
        args.layer_idx = [max_layers]
    else:
        layers = np.array(args.layer_idx)
        good = np.all((layers >= 0) & (layers <= max_layers))
        assert good, "Invalid layer number"

    return args


def select_tokenizer_and_model(args):

    model_name = args.embedding_type

    if model_name == "glove50":
        args.layer_idx = [1]
        return

    try:
        args.model, args.tokenizer = tfsemb_dwnld.download_tokenizers_and_models(
            model_name, local_files_only=True, debug=False
        )[model_name]
    except OSError:
        # NOTE: Please refer to make-target: cache-models for more information.
        print("Model and tokenizer not found. Please download into cache first.")
        exit(1)

    args = get_model_layer_count(args)

    if args.history and args.context_length <= 0:
        args.context_length = args.tokenizer.max_len_single_sentence
        assert (
            args.context_length <= args.tokenizer.max_len_single_sentence
        ), "given length is greater than max length"

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-type", type=str, default="glove")
    parser.add_argument("--context-length", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true", default=False)
    parser.add_argument("--save-hidden-states", action="store_true", default=False)
    parser.add_argument("--subject", type=str, default="625")
    parser.add_argument("--history", action="store_true", default=False)
    parser.add_argument("--conversation-id", type=int, default=0)
    parser.add_argument("--pkl-identifier", type=str, default=None)
    parser.add_argument("--project-id", type=str, default=None)
    parser.add_argument("--layer-idx", nargs="*", default=["all"])

    args = parser.parse_args()

    if len(args.layer_idx) == 1:
        if args.layer_idx[0].isdecimal():
            args.layer_idx = int(args.layer_idx[0])
        else:
            args.layer_idx = args.layer_idx[0]
    else:
        try:
            args.layer_idx = list(map(int, args.layer_idx))
        except ValueError:
            print("Invalid layer index")
            exit(1)

    return args


@main_timer
def main():
    args = parse_arguments()
    select_tokenizer_and_model(args)
    setup_environ(args)

    utterance_df = load_pickle(args)
    utterance_df = select_conversation(args, utterance_df)

    if len(utterance_df) == 0:
        print("Conversation data does not exist")
        return

    if args.embedding_type == "glove50":
        generate_func = generate_glove_embeddings
    elif args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        generate_func = generate_causal_embeddings
    elif args.embedding_type in tfsemb_dwnld.SEQ2SEQ_MODELS:
        generate_func = generate_conversational_embeddings
    else:
        generate_func = generate_embeddings

    output = generate_func(args, utterance_df)
    if len(output) == 2:
        df, embeddings = output
    else:
        df = output

    save_pickle(args, df, args.output_file, embeddings)

    return


if __name__ == "__main__":
    # NOTE: Before running this script please refer to the cache-models target
    # in the Makefile
    main()
