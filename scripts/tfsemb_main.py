import os
import tfsemb_download as tfsemb_dwnld
from tfsemb_config import parse_arguments, setup_environ
from tfsemb_genemb_causal import generate_causal_embeddings
from tfsemb_genemb_glove import generate_glove_embeddings
from tfsemb_genemb_mlm import generate_mlm_embeddings
from tfsemb_genemb_seq2seq import generate_conversational_embeddings
from tfsemb_genemb_whisper import generate_speech_embeddings
from tfsemb_genemb_mlm import generate_mlm_embeddings
from utils import load_pickle
from utils import save_pickle as svpkl


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
            item.to_pickle(filename)
    else:
        filename = file_name % args.layer_idx[0]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        item.to_pickle(filename)
    return


def select_conversation(args, df):
    if args.conversation_id:
        print("Selecting conversation", args.conversation_id)
        df = df[df.conversation_id == args.conversation_id]
        assert df.conversation_id.unique().shape[0] == 1
        print(
            f"Conversation Name: {df.conversation_name.unique()[0]}, shape: {df.shape}"
        )
    return df


def printe(example, args):
    tokenizer = args.tokenizer
    print(tokenizer.decode(example["encoder_ids"]))
    print(tokenizer.convert_ids_to_tokens(example["decoder_ids"]))
    print()


# @main_timer
def main():
    args = parse_arguments()
    setup_environ(args, "gen-emb")
    breakpoint()

    if os.path.exists(args.base_df_path):
        base_df = load_pickle(args.base_df_path)
    else:
        raise Exception("Base dataframe does not exist")

    utterance_df = select_conversation(args, base_df)
    print(
        args.conversation_id, utterance_df.conversation_name.unique(), len(utterance_df)
    )
    assert len(utterance_df) != 0, "Empty dataframe"

    # Select generation function based on model type
    match args.embedding_type:
        case "glove50":
            generate_func = generate_glove_embeddings
        case item if item in tfsemb_dwnld.CAUSAL_MODELS:
            generate_func = generate_causal_embeddings
        case item if item in tfsemb_dwnld.SEQ2SEQ_MODELS:
            generate_func = generate_conversational_embeddings
        case item if item in tfsemb_dwnld.SPEECHSEQ2SEQ_MODELS:
            generate_func = generate_speech_embeddings
        case item if item in tfsemb_dwnld.MLM_MODELS:
            generate_func = generate_mlm_embeddings
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
                is_dataframe=True,
            )
    else:
        df = output

    save_pickle(args, df, embeddings)

    return


if __name__ == "__main__":
    # NOTE: Before running this script please refer to the cache-models target
    # in the Makefile
    main()
