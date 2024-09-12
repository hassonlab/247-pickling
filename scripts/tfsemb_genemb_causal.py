import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
import tfsemb_download as tfsemb_dwnld


def make_input_from_tokens(args, token_list):
    size = args.context_length

    if len(token_list) <= size:
        windows = [tuple(token_list)]
    else:
        windows = [
            tuple(token_list[x : x + size]) for x in range(len(token_list) - size + 1)
        ]

    return windows


def make_dataloader_from_input(windows, batch_size):
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=batch_size, shuffle=False)
    return data_dl


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

    k = 250  # HACK (subject to change)
    top1_probabilities, top1_probabilities_idx = torch.topk(
        prediction_probabilities, k, dim=1
    )
    top1_probabilities, top1_probabilities_idx = (
        top1_probabilities.squeeze(),
        top1_probabilities_idx.squeeze(),
    )

    if k == 1:
        predicted_tokens = args.tokenizer.convert_ids_to_tokens(top1_probabilities_idx)
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
                    args.tokenizer.convert_tokens_to_string([token])
                    for token in token_list
                ]
                for token_list in predicted_tokens
            ]

    # top-1 probabilities
    top1_probabilities = [None] + top1_probabilities.tolist()
    # top-1 word
    top1_words = [None] + predicted_words
    # probability of correct word
    true_y_probability = [None] + prediction_probabilities.gather(1, true_y).squeeze(
        -1
    ).tolist()
    # true y rank
    vocab_rank = torch.argsort(prediction_probabilities, dim=-1, descending=True)
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
    # df_logits["logits"] = final_logits

    return df, df_logits, final_embeddings
