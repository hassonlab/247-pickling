import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import tfsemb_download as tfsemb_dwnld


def get_utt_info(df, ctx_len, multiple_convo=False):
    if multiple_convo:
        df["token_idx_in_sntnc"] = (
            df.groupby([df.conversation_id, df.sentence_idx]).cumcount() + 1
        )  # get token_idx in sentence
    else:
        df["token_idx_in_sntnc"] = (
            df.groupby(df.sentence_idx).cumcount() + 1
        )  # get token_idx in sentence

    df["num_tokens_in_sntnc"] = np.where(
        (df["sentence_idx"].ne(df["sentence_idx"].shift(-1)))
        | (df["conversation_id"].ne(df["conversation_id"].shift(-1))),
        df["token_idx_in_sntnc"],
        np.nan,
    )
    df["num_tokens_in_sntnc"] = (
        df["num_tokens_in_sntnc"].bfill().astype(int)
    )  # get # of tokens in sentence

    original_len = len(df.index)
    df = df.loc[df["num_tokens_in_sntnc"] <= ctx_len, :]
    new_len = len(df.index)
    if new_len < original_len:
        print(f"Deleted sentence, reducing {original_len - new_len} tokens ")

    return df


def get_mask_token_ids(args):
    mask_string = "[MASK]"
    if "roberta" in args.tokenizer.name_or_path:
        mask_string = "<mask>"
    special_tokens = args.tokenizer.encode(mask_string)

    return special_tokens


def make_input_from_tokens_utt(args, df):
    windows = df.groupby("sentence_idx")["token_id"].apply(tuple).tolist()
    if "bert" in args.tokenizer.name_or_path:
        special_tokens = args.tokenizer.encode("")
        windows = [
            (special_tokens[0],) + window + (special_tokens[1],) for window in windows
        ]
    # mask_ids = np.repeat(-1, len(windows)) # old mask id
    mask_ids = [tuple(range(1, len(window) - 1)) for window in windows]
    return windows, mask_ids


def make_input_from_tokens_utt_new(args, df):
    print("Filling to max length")
    df2 = df
    df2.reset_index(inplace=True)
    windows = []
    mask_ids = []

    for sentence in df2.sentence_idx.unique():
        sentence_window = tuple(df2.index[df2.sentence_idx == sentence])
        start_index = max(0, sentence_window[-1] - args.context_length + 1)

        # full input window
        window = tuple(df2.loc[start_index : sentence_window[-1], "token_id"])
        windows.append(window)
        # track which idx to extract embeddings
        mask_id = tuple(idx - start_index + 1 for idx in sentence_window)
        mask_ids.append(mask_id)

    # add [CLS] to start and [SEP] to end
    special_tokens = args.tokenizer.encode("")
    windows = [
        (special_tokens[0],) + window + (special_tokens[1],) for window in windows
    ]

    return windows, mask_ids


def make_input_from_tokens_mask(args, token_list, window_type):
    assert len(token_list) == len(window_type.index)

    special_tokens = get_mask_token_ids(args)

    windows = []
    mask_ids = np.empty([0], dtype=int)
    for i, _ in enumerate(token_list):
        window = (special_tokens[0],)  # start window
        if args.lctx:  # adding left context
            # print("Adding left context")
            window = window + tuple(
                token_list[i + 1 - window_type.loc[i, "token_idx_in_sntnc"] : i]
            )
        if args.masked:  # adding masked token
            window = window + (special_tokens[1],)
        else:  # adding unmasked current token
            window = window + (token_list[i],)
        mask_ids = np.append(mask_ids, len(window) - 1)
        if args.rctx:
            # print("Adding right context")
            window = window + tuple(
                token_list[
                    i
                    + 1 : i
                    + window_type.loc[i, "num_tokens_in_sntnc"]
                    - window_type.loc[i, "token_idx_in_sntnc"]
                    + 1
                ]
            )
        elif args.rctxp:
            # print("Adding partial right context")
            window = window + tuple(
                token_list[
                    i
                    + 1 : min(
                        i + 11,
                        i
                        + window_type.loc[i, "num_tokens_in_sntnc"]
                        - window_type.loc[i, "token_idx_in_sntnc"]
                        + 1,
                    )
                ]
            )
        window = window + (special_tokens[2],)
        windows.append(window)

    return windows, mask_ids


def extract_select_vectors_bert(mask_idx, array):
    if mask_idx != -1:
        x = array[:, mask_idx, :].clone()
    else:
        breakpoint()  # HACK Need this for faster implementation
        x = array[:, 1:-1, :].clone()

    return x


def extract_select_vectors_all_layers_bert(mask_idx, array, layers=None):
    array_actual = tuple(y.cpu() for y in array)

    all_layers_x = dict()
    for layer_idx in layers:
        array = array_actual[layer_idx]
        all_layers_x[layer_idx] = extract_select_vectors_bert(mask_idx, array)

    return all_layers_x


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


def process_extracted_logits_bert(args, concat_logits, sentence_token_ids):
    """Get the probability for the _correct_ word"""

    prediction_scores = torch.cat(concat_logits, axis=0)
    prediction_probabilities = F.softmax(prediction_scores, dim=1)

    logp = np.log2(prediction_probabilities)
    entropy = torch.sum(-prediction_probabilities * logp, dim=1).tolist()

    top1_probabilities, top1_probabilities_idx = prediction_probabilities.max(dim=1)
    predicted_tokens = args.tokenizer.convert_ids_to_tokens(top1_probabilities_idx)
    predicted_words = predicted_tokens
    if args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS:
        predicted_words = [
            args.tokenizer.convert_tokens_to_string(token) for token in predicted_tokens
        ]

    # top-1 probabilities
    top1_probabilities = top1_probabilities.tolist()
    # top-1 word
    top1_words = predicted_words
    # probability of correct word
    true_y = torch.tensor(sentence_token_ids).unsqueeze(-1)
    true_y_probability = prediction_probabilities.gather(1, true_y).squeeze(-1).tolist()
    vocab_rank = torch.argsort(prediction_probabilities, dim=-1, descending=True)
    true_y_rank = ((vocab_rank == true_y).nonzero(as_tuple=True)[1] + 1).tolist()

    return (
        top1_words,
        top1_probabilities,
        true_y_probability,
        true_y_rank,
        entropy,
    )


def model_forward_pass_bert(args, model_input, mask_ids):
    model = args.model
    device = args.device

    with torch.no_grad():
        model = model.to(device)
        model.eval()
        all_embeddings = []
        all_logits = []
        for batch_idx, batch in enumerate(model_input):
            mask_idx = mask_ids[batch_idx]
            batch = torch.tensor([batch])
            batch = batch.to(args.device)
            model_output = model(batch)
            logits = model_output.logits.cpu()
            if isinstance(mask_idx, int) or isinstance(
                mask_idx, np.int64
            ):  # one masked token
                embeddings = extract_select_vectors_all_layers_bert(
                    mask_idx, model_output.hidden_states, args.layer_idx
                )
                logits = extract_select_vectors_bert(mask_idx, logits)
                all_embeddings.append(embeddings)
                all_logits.append(logits)
            else:  # a full utterance or sentence
                for i in mask_idx:
                    embeddings = extract_select_vectors_all_layers_bert(
                        i, model_output.hidden_states, args.layer_idx
                    )
                    single_logits = extract_select_vectors_bert(i, logits)
                    all_embeddings.append(embeddings)
                    all_logits.append(single_logits)

    return all_embeddings, all_logits


def generate_mlm_embeddings(args, df):
    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    final_true_y_rank = []
    if args.project_id == "podcast":  # get sentence idx for podcast
        df.loc[:, "sentence_end"] = 0
        end_strings = "!?."
        for end_string in end_strings:
            df.loc[df.token == end_string, "sentence_end"] = 1
        df.loc[:, "sentence_idx"] = df.sentence_end.cumsum()
        df.loc[df.sentence_end != 1, "sentence_idx"] = 0
        df["sentence_idx"] = df.sentence_idx.replace(to_replace=0, method="bfill")
    df = get_utt_info(df, args.context_length)
    token_list = df["token_id"].tolist()

    if args.lctx and args.rctx and not args.masked:
        print("No Mask full utterance")
        model_input, mask_ids = make_input_from_tokens_utt(args, df)
        # model_input, mask_ids = make_input_from_tokens_utt_new(args, df)
    else:
        print("Masked")
        sntnc_info = df.loc[
            :, ("production", "token_idx_in_sntnc", "num_tokens_in_sntnc")
        ].reset_index()
        model_input, mask_ids = make_input_from_tokens_mask(
            args, token_list, sntnc_info
        )

    embeddings, logits = model_forward_pass_bert(args, model_input, mask_ids)
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
    ) = process_extracted_logits_bert(args, logits, token_list)
    final_top1_word.extend(top1_word)
    final_top1_prob.extend(top1_prob)
    final_true_y_prob.extend(true_y_prob)
    final_true_y_rank.extend(true_y_rank)

    if len(final_embeddings) > 1:
        # TODO concat all embeddings and return a dictionary
        # previous: np.concatenate(final_embeddings, axis=0)
        raise NotImplementedError
    else:
        final_embeddings = final_embeddings[0]

    df = pd.DataFrame(index=df.index)
    df["top1_pred"] = final_top1_word
    df["top1_pred_prob"] = final_top1_prob
    df["true_pred_prob"] = final_true_y_prob
    df["surprise"] = -df["true_pred_prob"] * np.log2(df["true_pred_prob"])
    df["entropy"] = entropy
    df.drop(columns=["utt", "utt_len", "utt_index"], errors="ignore")

    # TODO logits
    df_logits = pd.DataFrame()

    return df, df_logits, final_embeddings
