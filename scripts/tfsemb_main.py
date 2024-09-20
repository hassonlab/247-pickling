import os
import tfsemb_download as tfsemb_dwnld
from tfsemb_config import parse_arguments, setup_environ
from tfsemb_genemb_glove import generate_glove_embeddings
from tfsemb_genemb_symbolic import generate_symbolic_embeddings
from tfsemb_genemb_causal import generate_causal_embeddings
from tfsemb_genemb_mlm import generate_mlm_embeddings
from tfsemb_genemb_seq2seq import generate_conversational_embeddings
from tfsemb_genemb_whisper import generate_speech_embeddings
from tfsemb_genemb_mlm import generate_mlm_embeddings
from utils import load_pickle
from utils import save_pickle as svpkl


def save_results(args, df, embeddings, df_logits):
    """Write 'item' to 'file_name.pkl'"""

    if embeddings is not None:
        for layer_idx, embedding in embeddings.items():
            df["embeddings"] = embedding.tolist()
            filename = args.emb_df_path % layer_idx
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_pickle(filename)
    else:
        filename = args.emb_df_path % args.layer_idx[0]
        breakpoint()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_pickle(filename)

    if df_logits is not None and args.logits:  # save logits
        svpkl(
            df_logits,
            os.path.join(args.logits_path),
            is_dataframe=True,
        )
    return


def printe(example, args):
    tokenizer = args.tokenizer
    print(tokenizer.decode(example["encoder_ids"]))
    print(tokenizer.convert_ids_to_tokens(example["decoder_ids"]))
    print()


# @main_timer
def main():
    args = parse_arguments()
    setup_environ(args, "gen-emb")

    # Select utterance df
    assert os.path.exists(args.base_df_path), "Base dataframe does not exist"
    base_df = load_pickle(args.base_df_path)
    print("Selecting conversation", args.conv_id)
    utterance_df = base_df[base_df.conversation_id == args.conv_id]
    print(args.conv_id, utterance_df.conversation_name.unique(), len(utterance_df))
    assert len(utterance_df) != 0, "Empty dataframe"

    # Select generation function based on model type
    match args.emb:
        case "glove50":
            generate_func = generate_glove_embeddings
        case item if "symbolic" in item:
            generate_func = generate_symbolic_embeddings
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
    output = generate_func(args, utterance_df)
    df, embeddings, df_logits = output
    save_results(args, df, embeddings, df_logits)  # save results

    return


if __name__ == "__main__":
    # NOTE: Before running this script please refer to the cache-models target
    # in the Makefile
    main()
