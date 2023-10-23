import numpy as np
import pandas as pd
import whisper
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F
import tfsemb_download as tfsemb_dwnld


class AudioDataset(data.Dataset):
    def __init__(self, args, audio, conversation_df, transform=None):
        self.args = args
        self.conversation_df = conversation_df
        self.audio = audio
        self.transform = transform

    def __len__(self):
        return len(self.conversation_df.index)

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
    data_dl = data.DataLoader(input_dataset, batch_size=1, shuffle=False)
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
        for batch_idx, batch in enumerate(data_dl):
            input_features = batch["input_features"].to(args.device)
            decoder_input_ids = batch["context_token_ids"].to(args.device)
            window_num = batch["window_num"].item()
            utt_idx = batch["utt_idx"].item()
            chunk_idx = batch["chunk_idx"].item()
            print(
                f"Batch id: {batch_idx}, utt {utt_idx}, chunk {chunk_idx}, windows: {window_num}"
            )

            model_output = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            embeddings = extract_select_vectors_concat_all_layers(
                window_num,
                model_output.encoder_hidden_states,
                args.layer_idx,
            )

            all_embeddings.append(embeddings)

    return all_embeddings


def generate_acoustic_embeddings(args, df):
    print("Making acoustic embeddings, stop now if that's not what you want")

    # taking unique chunks
    conversation_df = df.drop_duplicates(subset=["utt_idx", "chunk_idx"]).copy()
    assert conversation_df.duplicated(subset=["utt_onset", "utt_offset"]).sum() == 0
    assert conversation_df.window_num.sum() == len(df)
    conversation_df.reset_index(drop=True, inplace=True)
    conversation_df["audio_onset"] = (conversation_df.utt_onset + 3000) / 512
    conversation_df["audio_offset"] = (conversation_df.utt_offset + 3000) / 512

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
    input_dataset = AudioDataset(args, audio, conversation_df)
    input_dl = make_dataloader_from_dataset(input_dataset)
    embeddings = speech_model_forward_pass(args, input_dl)

    embeddings = process_extracted_embeddings_all_layers(args, embeddings)
    for embeddings_layer in embeddings:
        assert len(df) == embeddings[embeddings_layer].shape[0]

    df = pd.DataFrame(index=df.index)

    return df, None, embeddings
