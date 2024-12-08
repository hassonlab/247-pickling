import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import tfsemb_download as tfsemb_dwnld


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
                [args.tokenizer.convert_tokens_to_string(token) for token in token_list]
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

    # TODO
    df_logits = pd.DataFrame()

    return df, df_logits, final_embeddings
