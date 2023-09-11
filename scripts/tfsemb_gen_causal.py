import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
import tfsemb_download as tfsemb_dwnld


def make_input_from_tokens(args, token_list):
    size = args.context_length
    windows = [
        tuple(token_list[max(0, idx - size) : idx])
        for idx, _ in enumerate(token_list)
    ]
    windows = windows[1:]

    return windows


def make_input_from_tokens2(args, token_list):
    size = args.context_length
    windows = [
        tuple(token_list[idx : min(idx + 20, len(token_list))])
        for idx, _ in enumerate(token_list)
    ]
    windows = windows[1:]

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

        all_sequences = []
        all_logits = []
        for batch_idx, batch in enumerate(data_dl):
            if batch_idx % 10 == 0:
                print(f"Batch ID: {batch_idx}")
            batch = torch.tensor([batch])
            batch = batch.to(args.device)
            model_output = model.generate(
                batch,
                min_length=batch.shape[1] + 20,
                max_new_tokens=20,
                pad_token_id=args.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            sequence = model_output.sequences.cpu()[0, batch.shape[1] :]

            logits = [score.cpu() for score in model_output.scores]
            logits = torch.cat(logits)

            all_sequences.append(sequence)
            all_logits.append(logits)

    return all_sequences, all_logits


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


def process_extracted_logits(args, concat_logits, true_ys):
    """Get the probability for the _correct_ word"""
    assert len(concat_logits) == len(true_ys), "Incorrect number of outputs"

    final_top1_words = []
    final_top1_prob = []
    final_true_y_prob = []
    final_true_y_rank = []
    # final_entropy = []

    for prediction_scores, true_y in zip(concat_logits, true_ys):
        prediction_probabilities = F.softmax(prediction_scores, dim=1)

        # logp = np.log2(prediction_probabilities)
        # entropy = torch.sum(-prediction_probabilities * logp, dim=1).tolist()

        (
            top1_probabilities,
            top1_probabilities_idx,
        ) = prediction_probabilities.max(dim=1)
        predicted_tokens = args.tokenizer.convert_ids_to_tokens(
            top1_probabilities_idx
        )
        predicted_words = [
            args.tokenizer.convert_tokens_to_string(token)
            for token in predicted_tokens
        ]

        # top-1 probabilities
        top1_probabilities = top1_probabilities.tolist()

        # top-1 word
        top1_words = predicted_words

        # probability of correct word
        if len(true_y) < prediction_probabilities.shape[0]:
            prediction_probabilities = prediction_probabilities[
                : len(true_y), :
            ]
        true_y = torch.tensor(true_y).unsqueeze(-1)
        true_y_probability = (
            prediction_probabilities.gather(1, true_y).squeeze(-1).tolist()
        )
        vocab_rank = torch.argsort(
            prediction_probabilities, dim=-1, descending=True
        )
        true_y_rank = (
            (vocab_rank == true_y).nonzero(as_tuple=True)[1] + 1
        ).tolist()

        final_top1_words.append(top1_words)
        final_top1_prob.append(top1_probabilities)
        final_true_y_prob.append(true_y_probability)
        final_true_y_rank.append(true_y_rank)
        # final_entropy.append(entropy)

    return (
        [None] + final_top1_words,
        [None] + final_top1_prob,
        [None] + final_true_y_prob,
        [None] + final_true_y_rank,
        # [None] + final_entropy,
    )


def generate_causal(args, df):
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    token_list = df["token_id"].tolist()
    model_input = make_input_from_tokens(args, token_list)
    _, logits = model_forward_pass(args, model_input)

    true_y_check = make_input_from_tokens2(args, token_list)
    (
        top1_word,
        top1_prob,
        true_y_prob,
        true_y_rank,
        # entropy,
    ) = process_extracted_logits(args, logits, true_y_check)

    df = pd.DataFrame(index=df.index)
    df["top1_pred"] = top1_word
    df["top1_pred_prob"] = top1_prob
    df["true_pred_prob"] = true_y_prob
    df["true_pred_rank"] = true_y_rank
    # df["entropy"] = entropy

    return df
