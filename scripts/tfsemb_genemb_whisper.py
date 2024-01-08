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
        self.audio = audio
        self.conversation_df = conversation_df
        self.transform = transform

    def __len__(self):
        return len(self.conversation_df.index)

    def __getitem__(self, idx):
        sampling_rate = 16000

        #####################################
        ## for full model or decoder- only ##
        #####################################
        if self.args.model_type == "full" or self.args.model_type == "de-only":

            # get word onset and offset
            chunk_offset = self.conversation_df.word_offset.iloc[idx]
            chunk_onset = np.max([0, (chunk_offset - 30)])

            chunk_data = self.audio[
                int(chunk_onset * sampling_rate) : int(chunk_offset * sampling_rate)
            ]

            # we don't need those since we are not extracting encoder hidden states
            start_windows = 0
            num_windows = 0

        ###########################
        ## for full-onset model  ##
        ###########################
        elif self.args.model_type == "full-onset":

            # get word onset
            word_onset = self.conversation_df.word_onset.iloc[idx]

            # change this if we want to do 20 ms per unit #TODO 
            # look at word onset + 252.5 ms (podcast) / 232.5 ms (tfs)
            if self.args.bin_type == "fixed-bin":
                if self.args.project_id == "podcast":
                    chunk_offset = word_onset + 0.2525 
                    num_windows = 11 
                elif self.args.project_id == "tfs":
                    chunk_offset = word_onset + 0.2325
                    num_windows = 10

            elif self.args.bin_type == "var-bin":
                
                chunk_offset = self.conversation_df.word_offset.iloc[idx]
                num_windows = self.conversation_df.num_windows.iloc[idx]

            # get start of 30 sec audio chunk
            chunk_onset = np.max([0, (chunk_offset - 30)])

            # we don't need those since we are not extracting encoder hidden states
            start_windows = 0
            num_windows = 0

        #########################
        ## for full model n-1 ##
        #########################
        elif self.args.model_type == "full-n-1":
            # get word onset and offset
            if idx == 0:
                chunk_offset = self.conversation_df.word_onset.iloc[idx]
            else:
                chunk_offset = self.conversation_df.word_offset.iloc[idx - 1]

            chunk_onset = np.max([0, (chunk_offset - 30)])
            chunk_data = self.audio[
                int(chunk_onset * sampling_rate) : int(chunk_offset * sampling_rate)
            ]

            # we don't need those since we are not extracting encoder hidden states
            start_windows = 0
            num_windows = 0

        ######################
        ## for encoder only ##
        ######################
        elif self.args.model_type == "en-only":

            # get word onset
            word_onset = self.conversation_df.word_onset.iloc[idx]

            # change this if we want to do 20 ms per unit #TODO 
            # look at word onset + 252.5 ms (podcast) / 232.5 ms (tfs)
            if self.args.bin_type == "fixed-bin":
                if self.args.project_id == "podcast":
                    chunk_offset = word_onset + 0.2525 
                    num_windows = 11 
                elif self.args.project_id == "tfs":
                    chunk_offset = word_onset + 0.2325
                    num_windows = 10

            elif self.args.bin_type == "var-bin":
                
                chunk_offset = self.conversation_df.word_offset.iloc[idx]
                num_windows = self.conversation_df.num_windows.iloc[idx]

                # for two conversations (676 conv 68 and 38) chunk_offset is smaller than 30 sec, but it falls within the last unit of 
                # the encoder, so there is no padding applied - therefore this #HACK
                if round(chunk_offset,2) == 30:
                    chunk_offset = 30

            # get start of 30 sec audio chunk
            chunk_onset = np.max([0, (chunk_offset - 30)])

            # if the audio chunk we're providing to Whisper is shorter than 30 sec, padding tokens will be added
            # here we calculate the start of the tokens that represent actual audio
            if chunk_offset < 30:
                start_windows = int(
                    (
                        ((self.conversation_df.word_onset[idx] - chunk_onset) * 1000) # get the time elapsed before word onset in msec (which will be equal to word onset, right?) TODO
                        - 7.5 # first encoder bin starts at -7.5 ms
                    )
                    // 20 # each bin represents 20 msec
                    + 3 # shift bins by 3 (why?) TODO
                )
            else:
                start_windows = 1500 - num_windows

        #######################################################
        ## for testing different ways of shuffling the audio ##
        #######################################################

        ##########
        # normal #
        ##########

        if self.args.shuffle_audio == "none":
            chunk_data = self.audio[
                int(chunk_onset * sampling_rate) : int(chunk_offset * sampling_rate)
            ]

        ######################
        # different shuffles #
        ######################

        elif self.args.shuffle_audio == "samples":
            # get audio until current word
            chunk1 = self.audio[
                int(chunk_onset * sampling_rate) : int(word_onset * sampling_rate)
            ]
            # shuffle samples
            chunk1 = np.random.shuffle(chunk1)
            # get current word audio
            chunk2 = self.audio[
                int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
            ]
            # concatenate
            chunk_data = np.append(chunk1, chunk2)

        elif self.args.shuffle_audio == "phonemes":
            # get audio until current word
            chunk1 = self.audio[
                int(chunk_onset * sampling_rate) : int(word_onset * sampling_rate)
            ]
            try:
                # shuffle phonemes
                # split into smaller chunks of size (phoneme)
                chunk1 = np.array_split(chunk1, int((word_onset - chunk_onset) * 20))
                # shuffle and concatenate
                random.shuffle(chunk1)
                chunk1 = np.concatenate(chunk1)
                # get current word audio
                chunk2 = self.audio[
                    int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
                ]
                # concatenate
                chunk_data = np.append(chunk1, chunk2)
            except:
                chunk_data = self.audio[
                    int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
                ]

        elif self.args.shuffle_audio == "words":
            # get audio until current word
            chunk1 = self.audio[
                int(chunk_onset * sampling_rate) : int(word_onset * sampling_rate)
            ]
            # TODO there is one case (676, conversation 52), where this does not work, because word_onset is smaller than 0
            try:
                # shuffle words
                # split into smaller chunks of size (word)
                chunk1 = np.array_split(chunk1, int((word_onset - chunk_onset) * 4))
                # shuffle and concatenate
                random.shuffle(chunk1)
                chunk1 = np.concatenate(chunk1)
                # get current word audio
                chunk2 = self.audio[
                    int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
                ]
                # concatenate
                chunk_data = np.append(chunk1, chunk2)
            except:
                chunk_data = self.audio[
                    int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
                ]

        elif self.args.shuffle_audio == "flip":
            # get audio until current word (or cutoff defined)
            chunk1 = self.audio[
                int(chunk_onset * sampling_rate) : int(
                    (word_onset - self.args.cutoff) * sampling_rate
                )
            ]
            try:
                # flip audio
                chunk1 = np.flip(chunk1)
                # get current word audio
                chunk2 = self.audio[
                    int((word_onset - self.args.cutoff) * sampling_rate) : int(
                        chunk_offset * sampling_rate
                    )
                ]
                # concatenate
                chunk_data = np.append(chunk1, chunk2)
            except:
                chunk_data = self.audio[
                    int(word_onset * sampling_rate) : int(chunk_offset * sampling_rate)
                ]

        ######################################
        ## DEBUG for writing audio to files ##
        ######################################
        # chunk_name = f"results/tfs/audio_segments/{self.args.subject}-{self.args.conversation_id}-audio_segment_{idx:03d}-{self.conversation_df.word.iloc[idx]}.wav"
        # wavfile.write(chunk_name, sampling_rate, chunk_data)

        # input features
        inputs = self.args.processor.feature_extractor(
            chunk_data, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = inputs.input_features

        # to give empty audio input (test for decoder only)
        # contrast with zero attention for decoder only
        # input_features = torch.zeros(1,80,3000)

        # to give random audio input (for testing / baseline)
        # input_features = torch.randn(1,80,3000)

        if self.args.project_id == "podcast" and idx == 0:
            context_tokens = self.conversation_df.iloc[:8][
                "token"
            ].tolist()  # to also include first 8 words that appear in the audio, but are cut off due to onset = NaN (only podcast) / check if that can be removed
            context_tokens.extend(
                self.conversation_df[
                    (self.conversation_df.word_onset >= chunk_onset)
                    & (self.conversation_df.word_offset <= chunk_offset)
                ]["token"].tolist()
            )
        else:
            # get all tokens within input window
            context_df = self.conversation_df[
                (self.conversation_df["word_onset"] >= chunk_onset)
                & (
                    self.conversation_df["word_onset"]
                    <= self.conversation_df.iloc[idx].word_onset
                )
            ]

            # sort on word onset
            context_df = context_df.sort_values("word_onset")

            if len(context_df.index) == 0:
                context_df = context_df.append(self.conversation_df.iloc[idx])

            # split words in context by production and comprehension
            if self.args.prod_comp_split:
                context_df["new_index"] = context_df.index

                if (
                    context_df.loc[context_df["new_index"] == idx, "speaker"].iloc[0]
                    == "Speaker1"
                ):  # speaker
                    context_df = context_df[context_df.speaker == "Speaker1"]
                elif (
                    context_df.loc[context_df["new_index"] == idx, "speaker"].iloc[0]
                    != "Speaker1"
                ):  # listener
                    context_df = context_df[context_df.speaker != "Speaker1"]

            context_tokens = context_df["token"].tolist()

        if self.args.shuffle_words == "flip":
            # get context_tokens until current word (or cutoff we define)
            # if 0 we go to -2 - excluding last word
            # if 1 we go to -3 excluding last two words ... and so on
            try:
                shuffle_context = context_tokens[: int(-self.args.cutoff - 1)]
                random.shuffle(shuffle_context)
                shuffle_context.extend(context_tokens[int(-self.args.cutoff - 1) :])
                context_tokens = shuffle_context

            except:
                context_tokens = context_tokens

        # TODO look into how this works in new transformers version (current version: 4.23.1)
        prefix_tokens = self.args.tokenizer.tokenize(
            "<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>"
        )

        # prefix_tokens = self.args.tokenizer.tokenize("<|startoftranscript|> <|notimestamps|>")

        tokens = prefix_tokens + context_tokens

        token_ids = self.args.processor.tokenizer.encode(
            tokens, add_special_tokens=False, return_tensors="pt"
        )

        sample = {
            "input_features": input_features.squeeze(),
            "token_ids": token_ids.squeeze(),
            "start_windows": start_windows,
            "num_windows": num_windows
        }

        return sample


def make_dataloader_from_dataset(input_dataset):
    data_dl = data.DataLoader(input_dataset, batch_size=1, shuffle=False)
    return data_dl


def process_extracted_embeddings(args, concat_output):
    """(batch_size, max_len, embedding_size)"""

    embeddings = []

    for c in concat_output:

        embeddings.append(c.numpy().squeeze())

    return embeddings


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
            if "whisper" in args.embedding_type:
                true_y = torch.cat([sti[0, :], sti[1:, -1]]).unsqueeze(-1)
            else:
                true_y = torch.cat([sti[0, 1:], sti[1:, -1]]).unsqueeze(-1)

    prediction_probabilities = F.softmax(prediction_scores, dim=1)

    logp = np.log2(prediction_probabilities)
    entropy = [None] + torch.sum(-prediction_probabilities * logp, dim=1).tolist()

    top1_probabilities, top1_probabilities_idx = prediction_probabilities.max(dim=1)
    predicted_tokens = args.tokenizer.convert_ids_to_tokens(top1_probabilities_idx)
    predicted_words = predicted_tokens
    if (
        args.embedding_type in tfsemb_dwnld.CAUSAL_MODELS
        or tfsemb_dwnld.SPEECHSEQ2SEQ_MODELS
    ):
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


def extract_select_vectors(batch_idx, array):

    # what is going on here? #TODO
    # here we select either n or n-1 token
    # it would be better to use a param here instead of manually changing the code
    if batch_idx == 0:
        x = array[0, :-1, :].clone()
        if array.shape[0] > 1:
            rem_sentences_preds = array[1:, -1, :].clone()

            x = torch.cat([x, rem_sentences_preds], axis=0)
    else:
        try: # why do we have a try and except statement here? #TODO
            x = array[:, -1, :].clone()
        except:
            x = array[:, -2, :].clone()

    return x

def extract_select_vectors_concat(num_windows, start_windows, array):
    # concatenating all windows from start_windows to start_windows + num_windows
    x = array[:, start_windows, :]

    for i in range(1, num_windows):
        x = torch.cat((x, array[:, start_windows + i, :]), 1)

    return x


def extract_select_vectors_concat_all_layers(
    num_windows, start_windows, array, layers=None
):
    array_actual = tuple(y.cpu() for y in array)

    all_layers_x = dict()
    for layer_idx in layers:
        array = array_actual[layer_idx]
        all_layers_x[layer_idx] = extract_select_vectors_concat(
            num_windows, start_windows, array
        )

    return all_layers_x


def extract_select_vectors_logits(batch_idx, array):
    # TODO harsha please clean this up
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


def speech_model_forward_pass(args, data_dl):
    model = args.model
    device = args.device

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        all_embeddings = []
        all_logits = []
        for batch_idx, batch in enumerate(data_dl):

            # DEBUG
            print(batch_idx)

            input_features = batch["input_features"].to(args.device)
            decoder_input_ids = batch["token_ids"].to(args.device)
            start_windows = batch["start_windows"].item()
            num_windows = batch["num_windows"].item()

            #####################
            ## for full model: ##
            #####################
            if args.model_type == "full" or args.model_type == "full-onset":
                model_output = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                )
                logits = model_output.logits.cpu()
                embeddings = extract_select_vectors_all_layers(
                    batch_idx + 1, model_output.decoder_hidden_states, args.layer_idx
                )

            #####################
            ## for encoder only: ##
            #####################
            elif args.model_type == "en-only":
                if args.project_id == "podcast":
                    if args.bin_type == "fixed_bin":
                        num_windows = 12 
                    elif args.bin_type == "var-bin":
                        pass
                elif args.project_id == "tfs":
                    if args.bin_type == "fixed_bin":
                        num_windows = 10 
                    elif args.bin_type == "var-bin":
                        pass

                model_output = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                )
                logits = model_output.logits.cpu()
                embeddings = extract_select_vectors_concat_all_layers(
                    num_windows,
                    start_windows,
                    model_output.encoder_hidden_states,
                    args.layer_idx,
                )

            #######################
            ## for decoder only ##
            ######################
            elif args.model_type == "de-only":

                # set encoder_outputs to 0
                # TODO make it work for different model sizes
                if "tiny" in args.embedding_type:
                    encoder_outputs = torch.zeros(1, 1500, 384).to(args.device)
                elif "medium" in args.embedding_type:
                    encoder_outputs = torch.zeros(1, 1500, 1024).to(args.device)

                # set cross attention heads to 0
                cross_attn_head_mask = torch.zeros(
                    args.model.config.decoder_layers,
                    args.model.config.decoder_attention_heads,
                ).to(args.device)

                model_output = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True,
                    cross_attn_head_mask=cross_attn_head_mask,
                    encoder_outputs=encoder_outputs,
                )
                logits = model_output.logits.cpu()
                embeddings = extract_select_vectors_all_layers(
                    batch_idx + 1, model_output.decoder_hidden_states, args.layer_idx
                )

            # concatenate logits across batches
            logits = extract_select_vectors_logits(batch_idx, logits)

            all_embeddings.append(embeddings)
            all_logits.append(logits)

    return all_embeddings, all_logits


def generate_speech_embeddings(args, df):

    final_embeddings = []
    final_top1_word = []
    final_top1_prob = []
    final_true_y_prob = []
    final_true_y_rank = []
    final_logits = []

    # # disable pandas warnings #HACK
    # # change the code so that the warnings are not issued #TODO
    # pd.options.mode.chained_assignment = None 

    # drop rows that have NaN as onset or offset (only relevant for podcast)
    conversation_df = df.dropna(subset=['onset','offset']).copy()
    conversation_df.reset_index(drop=True, inplace=True)

    # translate onset and offset from Hz to sec and calculate word length
    conversation_df["word_onset"] = (conversation_df.onset + 3000) / 512
    conversation_df["word_offset"] = (conversation_df.offset + 3000) / 512
    conversation_df["word_len"] = conversation_df.word_offset - conversation_df.word_onset

    # calculate number of encoder windows to cover word length audio segment
    # explain formula #TODO
    conversation_df["num_windows"] = ((conversation_df["word_len"] - 0.0525) // 0.020 + 2).astype(int)
    conversation_df.loc[conversation_df["num_windows"] < 1, "num_windows"] = 1

    if args.project_id == "podcast":
        audio_path = "/scratch/gpfs/ln1144/247-pickling/data/podcast/podcast_16k.wav" #TODO change this to be flexible
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
    embeddings, logits = speech_model_forward_pass(args, input_dl)

    token_ids = conversation_df["token_id"].tolist()
    token_ids = [tuple(token_ids)]
    embeddings = process_extracted_embeddings_all_layers(args, embeddings)
    for _, item in embeddings.items():
        assert len(item) == len(token_ids[0])
    final_embeddings.append(embeddings)

    (
        top1_word,
        top1_prob,
        true_y_prob,
        true_y_rank,
        entropy,
    ) = process_extracted_logits(args, logits, token_ids)

    final_top1_word.extend(top1_word)
    final_top1_prob.extend(top1_prob)
    final_true_y_prob.extend(true_y_prob)
    final_true_y_rank.extend(true_y_rank)
    final_logits.extend(torch.cat(logits, axis=0).tolist())

    if len(final_embeddings) > 1:
        # TODO concat all embeddings and return a dictionary
        # previous: np.concatenate(final_embeddings, axis=0)
        raise NotImplementedError
    else:
        final_embeddings = final_embeddings[0]

    # HACK
    # None is added as first item (why?)
    # therefore we select from index 1 onwards
    df = pd.DataFrame()
    df["top1_pred"] = final_top1_word[1:]
    df["top1_pred_prob"] = final_top1_prob[1:]
    df["true_pred_prob"] = final_true_y_prob[1:]
    df["true_pred_rank"] = final_true_y_rank[1:]
    df["surprise"] = -df["true_pred_prob"] * np.log2(df["true_pred_prob"])
    df["entropy"] = entropy[1:]

    df_logits = pd.DataFrame()

    print('Conversation ' + str(args.conversation_id) + ' done. Prediction accuracy: ' + str(np.mean(df.true_pred_rank == 1)))

    return df, df_logits, final_embeddings