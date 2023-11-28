import os
import pickle
import sys

import gensim.downloader as api
import numpy as np
import pandas as pd
import tfspickling.tfsemb_download as tfsemb_dwnld
import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
from tfspickling.tfsemb_config import setup_environ
from tfspickling.tfsemb_parser import arg_parser
from tfspickling.utils import load_pickle, main_timer
from tfspickling.utils import save_pickle as svpkl


def save_pickle(args, item, embeddings=None):
    """Write 'item' to 'file_name.pkl'"""
    file_name = args.output_file
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


def check_token_is_root(args, df):
    token_is_root_string = args.embedding_type.split("/")[-1] + "_token_is_root"
    df[token_is_root_string] = (
        df["word"]
        == df["token"]
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
    )

    return df


def convert_token_to_idx(args, df):
    df["token_id"] = df["token"].apply(args.tokenizer.convert_tokens_to_ids)
    return df


def convert_token_to_word(args, df):
    assert "token" in df.columns, "token column is missing"

    df["token2word"] = (
        df["token"]
        .apply(args.tokenizer.convert_tokens_to_string)
        .str.strip()
        .str.lower()
    )
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
    df = df.explode("token", ignore_index=False)
    df = convert_token_to_word(args, df)
    df = convert_token_to_idx(args, df)
    df = check_token_is_root(args, df)

    df["token_idx"] = (
        df.groupby(["adjusted_onset", "word"]).cumcount()
    ).astype(int)
    df = df.reset_index(drop=True)

    return df


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
        layer_embeddings[layer_idx] = process_extracted_embeddings(
            args, concat_output
        )

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
    entropy = [None] + torch.sum(
        -prediction_probabilities * logp, dim=1
    ).tolist()

    k = 250  # HACK (subject to change)
    top1_probabilities, top1_probabilities_idx = torch.topk(
        prediction_probabilities, k, dim=1
    )
    top1_probabilities, top1_probabilities_idx = (
        top1_probabilities.squeeze(),
        top1_probabilities_idx.squeeze(),
    )

    if k == 1:
        predicted_tokens = args.tokenizer.convert_ids_to_tokens(
            top1_probabilities_idx
        )
    else:
        predicted_tokens = [
            args.tokenizer.convert_ids_to_tokens(item)
            for item in top1_probabilities_idx
        ]

    predicted_words = predicted_tokens
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        if k == 1:
            predicted_words = [
                args.tokenizer.convert_tokens_to_string(token)
                for token in predicted_tokens
            ]
        else:
            predicted_words = [
                [
                    args.tokenizer.convert_tokens_to_string(token)
                    for token in token_list
                ]
                for token_list in predicted_tokens
            ]

    # top-1 probabilities
    top1_probabilities = [None] + top1_probabilities.tolist()
    # top-1 word
    top1_words = [None] + predicted_words
    # probability of correct word
    true_y_probability = [None] + prediction_probabilities.gather(
        1, true_y
    ).squeeze(-1).tolist()
    # true y rank
    vocab_rank = torch.argsort(
        prediction_probabilities, dim=-1, descending=True
    )
    true_y_rank = [None] + (
        (vocab_rank == true_y).nonzero(as_tuple=True)[1] + 1
    ).tolist()

    # TODO: probabilities of all words

    return (
        top1_words,
        top1_probabilities,
        true_y_probability,
        true_y_rank,
        entropy,
    )


def extract_select_vectors(batch_idx, array):
    if batch_idx == 0:  # first batch
        x = array[0, :-1, :].clone()  # first window, all but last embeddings
        if array.shape[0] > 1:
            try:  # (n-1)-th embedding
                rem_sentences_preds = array[1:, -2, :].clone()
            except:  # n-th embedding
                rem_sentences_preds = array[1:, -1, :].clone()

            x = torch.cat([x, rem_sentences_preds], axis=0)
    else:  # remaining batches
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
            if batch_idx % 10 == 0:
                print(f"Batch ID: {batch_idx}")
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
                input_ids.unsqueeze(0),
                decoder_input_ids=decoder_ids.unsqueeze(0),
            )
            # After: get all relevant layers
            embeddings = {
                i: outputs[decoderkey][i - 8].cpu()[0, :-1, :]
                for i in decoderlayers
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
                            encoder_embs[i][-token_idx - 1] = outputs[
                                encoderkey
                            ][i][
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
            examples.append(
                {"encoder_ids": context, "decoder_ids": response[:-1]}
            )

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
        top1_word, top1_prob, true_y_prob, _, _ = process_extracted_logits(
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
            tuple(token_list[x : x + size])
            for x in range(len(token_list) - size + 1)
        ]

    return windows


def make_dataloader_from_input(windows, batch_size):
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=batch_size, shuffle=False)
    return data_dl


def inference_function(args, model_input):
    accelerator = Accelerator()

    @find_executable_batch_size(starting_batch_size=128)
    def inner_training_loop(batch_size=128):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        accelerator.print(f"Trying batch size: {batch_size}")
        input_dl = make_dataloader_from_input(model_input, batch_size)
        embeddings, logits = model_forward_pass(args, input_dl)

        return embeddings, logits

    embeddings, logits = inner_training_loop()

    return embeddings, logits


def generate_causal_embeddings(args, df):
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        args.tokenizer.pad_token = args.tokenizer.eos_token
    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    final_true_y_rank = []
    final_logits = []

    token_list = df["token_id"].tolist()
    model_input = make_input_from_tokens(args, token_list)
    embeddings, logits = inference_function(args, model_input)

    embeddings = process_extracted_embeddings_all_layers(args, embeddings)
    for _, item in embeddings.items():
        assert item.shape[0] == len(token_list)
    final_embeddings.append(embeddings)

    (
        top1_word,
        top1_prob,
        true_y_prob,
        true_y_rank,
        entropy,
    ) = process_extracted_logits(args, logits, model_input)
    final_top1_word.extend(top1_word)
    final_top1_prob.extend(top1_prob)
    final_true_y_prob.extend(true_y_prob)
    final_true_y_rank.extend(true_y_rank)
    final_logits.extend([None] + torch.cat(logits, axis=0).tolist())

    if len(final_embeddings) > 1:
        # TODO concat all embeddings and return a dictionary
        # previous: np.concatenate(final_embeddings, axis=0)
        raise NotImplementedError
    else:
        final_embeddings = final_embeddings[0]

    df = pd.DataFrame(index=df.index)
    df["topk_pred"] = final_top1_word
    df["topk_pred_prob"] = final_top1_prob
    df["true_pred_prob"] = final_true_y_prob
    df["true_pred_rank"] = final_true_y_rank
    df["surprise"] = -df["true_pred_prob"] * np.log2(df["true_pred_prob"])
    df["entropy"] = entropy

    df_logits = pd.DataFrame()
    df_logits["logits"] = final_logits

    return df, df_logits, final_embeddings


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def generate_glove_embeddings(args, df):
    df1 = pd.DataFrame()
    glove = api.load("glove-wiki-gigaword-50")
    df1["embeddings"] = df["word"].apply(lambda x: get_vector(x.lower(), glove))

    return df1


# @main_timer
def main():
    args = arg_parser()
    setup_environ(args)

    if os.path.exists(args.base_df_file):
        base_df = load_pickle(args.base_df_file)
    else:
        raise Exception("Base dataframe does not exist")

    utterance_df = select_conversation(args, base_df)
    assert len(utterance_df) != 0, "Empty dataframe"

    # Select generation function based on model type
    match args.embedding_type:
        case "glove50":
            generate_func = generate_glove_embeddings
        case item if item in tfsemb_dwnld.CAUSAL_MODELS:
            generate_func = generate_causal_embeddings
        case item if item in tfsemb_dwnld.SEQ2SEQ_MODELS:
            generate_func = generate_conversational_embeddings
        case _:
            print('Invalid embedding type: "{}"'.format(args.embedding_type))
            exit()

    # Generate Embeddings
    embeddings = None
    output = generate_func(args, utterance_df)
    if len(output) == 3:
        df, df_logits, embeddings = output
        if not df_logits.empty:
            svpkl(
                df_logits,
                os.path.join(args.logits_folder, args.output_file_name),
            )
    else:
        df = output

    save_pickle(args, df, embeddings)

    return


if __name__ == "__main__":
    # NOTE: Before running this script please refer to the cache-models target
    # in the Makefile
    main()
