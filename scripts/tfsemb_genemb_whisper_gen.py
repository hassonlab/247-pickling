import numpy as np
import pandas as pd
import whisper
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F
import tfsemb_download as tfsemb_dwnld

WHISPER_XATN_HEADS = (
    {  # https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a
        "openai/whisper-tiny.en": [
            [1, 0],
            [2, 0],
            [2, 5],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ],
        "openai/whisper-large": [
            [9, 19],
            [11, 2],
            [11, 4],
            [11, 17],
            [22, 7],
            [22, 11],
            [22, 17],
            [23, 2],
            [23, 15],
        ],
        "openai/whisper-large-v2": [
            [10, 12],
            [13, 17],
            [16, 11],
            [16, 12],
            [16, 13],
            [17, 15],
            [17, 16],
            [18, 4],
            [18, 11],
            [18, 19],
            [19, 11],
            [21, 2],
            [21, 3],
            [22, 3],
            [22, 9],
            [22, 12],
            [23, 5],
            [23, 7],
            [23, 13],
            [25, 5],
            [26, 1],
            [26, 12],
            [27, 15],
        ],
    }
)


class AudioDataset(data.Dataset):
    def __init__(self, args, audio, conversation_df, transform=None):
        self.args = args
        self.audio = audio

    def __getitem__(self, idx):
        sampling_rate = 16000

        ######################
        ## for encoder only ##
        ######################
        assert self.args.model_type == "en-only"

        chunk_onset = self.conversation_df.audio_onset.iloc[idx]
        chunk_offset = self.conversation_df.audio_offset.iloc[idx]

        # generate input features
        chunk_data = self.audio[
            int(chunk_onset * sampling_rate) : int(chunk_offset * sampling_rate)
        ]
        inputs = self.args.processor.feature_extractor(
            chunk_data, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = inputs.input_features

        # generate decoder tokens
        prefix_tokens = self.args.tokenizer.tokenize(
            "<|startoftranscript|> <|en|> <|transcribe|>"
        )
        prefix_tokens = self.args.tokenizer.tokenize("<|startoftranscript|> ")
        context_token_ids = self.args.processor.tokenizer.encode(
            prefix_tokens, add_special_tokens=False, return_tensors="pt"
        )

        sample = {
            "input_features": input_features.squeeze(),
            "context_token_ids": context_token_ids.squeeze(),
            "utt_idx": self.conversation_df.utt_idx.iloc[idx],
            "chunk_idx": self.conversation_df.chunk_idx.iloc[idx],
            "window_num": self.conversation_df.window_num.iloc[idx],
        }

        return sample


def make_dataloader_from_dataset(input_dataset):
    data_dl = data.DataLoader(input_dataset, shuffle=False)
    return data_dl


def process_extracted_embeddings(args, concat_output):
    """(batch_size, max_len, embedding_size)"""

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


def extract_select_vectors_concat(num_windows, array):
    # concatenating all windows from 0 to num_windows
    x = array[0, 0:num_windows, :]
    return x


def extract_select_vectors_concat_all_layers(num_windows, array, layers=None):
    array_actual = tuple(y.cpu() for y in array)

    all_layers_x = dict()
    for layer_idx in layers:
        array = array_actual[layer_idx]
        all_layers_x[layer_idx] = extract_select_vectors_concat(num_windows, array)

    return all_layers_x


def speech_model_forward_pass(args, data_dl):
    model = args.model
    device = args.device

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        all_embeddings = []
        breakpoint()
        # model.generation_config.alignment_heads = WHISPER_XATN_HEADS[
        #     args.embedding_type
        # ]
        for batch_idx, batch in enumerate(data_dl):
            input_features = batch["input_features"].to(args.device)
            print(f"Batch ID: {batch_idx}")

            model_output = model.generate(
                input_features=input_features,
                max_new_tokens=1000,
                return_token_timestamps=True,  # dtw timestamp
            )
            print(args.tokenizer.decode(model_output.sequences[0]))
            print(model_output.token_timestamps)
            # embeddings = extract_select_vectors_concat_all_layers(
            #     window_num,
            #     model_output.encoder_hidden_states,
            #     args.layer_idx,
            # )

            # all_embeddings.append(embeddings)
        breakpoint()

    return all_embeddings


def generate_whisper_gen_embeddings(args, df):
    print("Generating whisper embeddings using generate")

    if args.project_id == "podcast":
        audio_path = "/scratch/gpfs/ln1144/247-pickling/data/podcast/podcast_16k.wav"
    elif args.project_id == "tfs":
        audio_path = (
            "data/"
            + str(args.project_id)
            + "/"
            + str(args.subject)
            + "/"
            + df.conversation_name.unique().item()
            + "/audio/"
            + df.conversation_name.unique().item()
            + "_deid.wav"
        )

    audio = whisper.load_audio(audio_path)

    # chunking audio to 30s
    chunk_inputs = []
    chunk_start_idx = 0
    sampling_rate = 16000
    while chunk_start_idx < len(audio):
        new_chunk_start_idx = chunk_start_idx + 30 * sampling_rate
        chunk = audio[chunk_start_idx:new_chunk_start_idx]
        inputs = args.processor.feature_extractor(
            chunk, return_tensors="pt", sampling_rate=sampling_rate
        )
        chunk_inputs.append(inputs)
        chunk_start_idx = new_chunk_start_idx

    embeddings = speech_model_forward_pass(args, chunk_inputs)

    embeddings = process_extracted_embeddings_all_layers(args, embeddings)
    for embeddings_layer in embeddings:
        assert len(df) == embeddings[embeddings_layer].shape[0]

    df = pd.DataFrame(index=df.index)

    return df, None, embeddings
