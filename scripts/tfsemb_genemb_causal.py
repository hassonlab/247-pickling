import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.utils.data as data
from accelerate import Accelerator, find_executable_batch_size
import tfsemb_download as tfsemb_dwnld


def make_input_from_tokens(context_len, token_list):
    """Construct input batches to LLM based on one token list

    Args:
        context_len (int): context_length
        token_list (list): list of token_ids

    Returns:
        windows (list): list of tuples of inputs token_ids
    """

    if len(token_list) <= context_len:
        windows = [tuple(token_list)]
    else:
        windows = [
            tuple(token_list[x : x + context_len])
            for x in range(len(token_list) - context_len + 1)
        ]

    return windows


def make_embedding_input(args, df):
    """Add chunk_idx column to split df into chunks

    Args:
        args (namespace): configuration
        df (df): datum

    Returns:
        df (df): datum
    """
    FS = 512  # sample rate of ECoG
    if args.context_level == "part":  # part level
        df = df.assign(chunk_idx=1)
    elif args.context_level == "conversation":  # convo level
        df = df.assign(
            chunk_idx=df.onset - df.offset.shift() >= FS * 60 * args.convo_cutoff
        )  # Check the gap between words to compare with the convo cutoff (in minutes)
        df.chunk_idx = df.chunk_idx.cumsum()
    elif args.context_level == "utterance":  # utterance level
        df = df.assign(chunk_idx=df.speaker.ne(df.speaker.shift()))
        df.chunk_idx = df.chunk_idx.cumsum()

    return df


def extract_select_vectors(batch_idx, array, emb_type):
    """Extract one layer of results from model outputs"""
    assert array.shape[0] == 1, "Something weird with batch"
    if batch_idx == 0:  # first batch (take everything)
        if emb_type == "n-1":
            x = array[0, :-1, :].clone()  # all but last embeddings
        elif emb_type == "n":
            x = array[0, :, :].clone()  # everything

    else:  # remaining batches
        if emb_type == "n-1":
            try:
                x = array[0, -2, :].clone()
            except:  # if only one word or something, then can't take n-1
                x = None
        elif emb_type == "n":
            x = array[0, -1, :].clone()

    return x


def extract_select_vectors_all_layers(args, batch_idx, array):
    """Extract layers of results from model outputs"""
    array_actual = tuple(y.cpu() for y in array)

    all_layers_x = dict()
    for layer_idx in args.layer_idx:
        array = array_actual[layer_idx]
        all_layers_x[layer_idx] = extract_select_vectors(
            batch_idx, array, args.emb_type
        )

    return all_layers_x


def make_dataloader_from_input(windows, batch_size):
    """Make batched dataloader from input token_ids"""
    input_ids = torch.tensor(windows)
    data_dl = data.DataLoader(input_ids, batch_size=batch_size, shuffle=False)
    return data_dl


def model_forward_pass(args, data_dl):
    """Embedding generation

    Args:
        args (namespace): configuration
        data_dl (dataloader): dataloader for input token_ids

    Returns:
    """
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
            batch = batch.to(device)
            model_output = model(batch)

            embeddings = extract_select_vectors_all_layers(
                args, batch_idx, model_output.hidden_states
            )
            logits = model_output.logits.cpu()
            logits = extract_select_vectors(batch_idx, logits, "n-1")

            all_embeddings.append(embeddings)
            all_logits.append(logits)

    return all_embeddings, all_logits


def inference_function(args, model_input):
    """Embedding generation batched

    Args:
        args (namespace): configuration
        model_input (list): list of tuples of inputs tokens

    Returns:
    """

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
    """Process and concatenate embeddings from different batches
    (batch_size, max_len, embedding_size)
    """
    # concatenate all batches
    concatenated_embeddings = torch.cat(concat_output, dim=0).numpy()
    extracted_embeddings = concatenated_embeddings

    if args.emb_type == "n-1":
        emb_dim = concatenated_embeddings.shape[-1]

        # the first token is always empty
        init_token_embedding = np.empty((1, emb_dim)) * np.nan

        extracted_embeddings = np.concatenate(
            [init_token_embedding, concatenated_embeddings], axis=0
        )

    return extracted_embeddings


def process_extracted_embeddings_all_layers(args, layer_embeddings_dict):
    """Process and concatenate embeddings from different batches"""
    layer_embeddings = dict()
    for layer_idx in args.layer_idx:
        concat_output = []
        for item_dict in layer_embeddings_dict:
            concat_output.append(item_dict[layer_idx])
        layer_embeddings[layer_idx] = process_extracted_embeddings(args, concat_output)

    return layer_embeddings


def process_extracted_logits(args, concat_logits, sentence_token_ids):
    """Extract topk probabilities and predictions, as well true y rank and prob

    Args:
        args (namespace): configuration
        concat_logits (list): list of torch tensors of shape (num_token, num_dictionary)
        sentence_token_ids (list): list of input token_ids

    Returns:

    """
    # (batch_size, max_len, vocab_size)

    # logits
    prediction_scores = torch.cat(concat_logits, axis=0)  # concat all batches
    prediction_probabilities = F.softmax(prediction_scores.float(), dim=1)  # softmax
    logp = np.log2(prediction_probabilities)  # log prob
    entropy = [None] + torch.sum(-prediction_probabilities * logp, dim=1).tolist()
    prediction_scores = [None] + prediction_scores.tolist()

    # Top probabilities
    topk_probabilities, topk_probabilities_idx = torch.topk(
        prediction_probabilities, args.topk, dim=1
    )
    topk_probabilities = [None] + topk_probabilities.tolist()  # top k probs

    # Top words
    if args.topk == 1:
        predicted_tokens = args.tokenizer.convert_ids_to_tokens(topk_probabilities_idx)
        predicted_words = [
            args.tokenizer.convert_tokens_to_string([token])
            for token in predicted_tokens
        ]
    else:
        predicted_tokens = [
            args.tokenizer.convert_ids_to_tokens(item)
            for item in topk_probabilities_idx
        ]
        predicted_words = [
            [args.tokenizer.convert_tokens_to_string([token]) for token in token_list]
            for token_list in predicted_tokens
        ]
    topk_words = [None] + predicted_words  # top k words

    # True y probability
    true_y = torch.tensor(sentence_token_ids).unsqueeze(-1)
    true_y = true_y[1:, :]
    true_y_probability = [None] + prediction_probabilities.gather(1, true_y).squeeze(
        -1
    ).tolist()  # true y prob

    # True y rank
    vocab_rank = torch.argsort(prediction_probabilities, dim=-1, descending=True)
    true_y_rank = [None] + (
        (vocab_rank == true_y).nonzero(as_tuple=True)[1] + 1
    ).tolist()  # true y rank

    return (
        prediction_scores,
        topk_words,
        topk_probabilities,
        true_y_probability,
        true_y_rank,
        entropy,
    )


def generate_causal_embeddings(args, df):
    """Generate embeddings for causal models

    Args:
        args (namespace): configuration
        df (df): dataframe of tokens

    Returns:

    """
    if args.tokenizer.pad_token is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    all_embeddings = {}
    all_logits = []
    df_all = pd.DataFrame()

    # Extra embeddings and logits
    df = make_embedding_input(args, df)

    for chunk_idx, subdf in df.groupby(
        "chunk_idx", axis=0
    ):  # loop through chunks (convo/utt)
        print(f"Extracting for {args.context_level} {chunk_idx}")

        # Extract embeddings and logits
        token_list = subdf["token_id"].tolist()
        model_input = make_input_from_tokens(args.context_length, token_list)
        embeddings, logits = inference_function(args, model_input)

        # Process embeddings
        embeddings = process_extracted_embeddings_all_layers(args, embeddings)
        if len(all_embeddings) == 0:
            all_embeddings = embeddings
        else:  # concate embeddings from previous chunks
            for key, item in embeddings.items():
                all_embeddings[key] = np.concatenate((all_embeddings[key], item))

        # Process logits
        (
            logits,
            topk_word,
            topk_prob,
            true_y_prob,
            true_y_rank,
            entropy,
        ) = process_extracted_logits(args, logits, token_list)
        all_logits.extend(logits)
        breakpoint()

        # Organize and save
        subdf = pd.DataFrame(index=subdf.index)
        subdf["topk_pred"] = topk_word
        subdf["topk_pred_prob"] = topk_prob
        subdf["true_pred_prob"] = true_y_prob
        subdf["true_pred_rank"] = true_y_rank
        subdf["surprise"] = -subdf["true_pred_prob"] * np.log2(subdf["true_pred_prob"])
        subdf["entropy"] = entropy
        df_all = pd.concat((df_all, subdf))

    breakpoint()
    for _, item in all_embeddings.items():
        assert item.shape[0] == len(df)
    breakpoint()

    df_logits = pd.DataFrame()
    if args.logits:  # save logits
        df_logits["logits"] = all_logits
    breakpoint()

    return df, embeddings, df_logits
