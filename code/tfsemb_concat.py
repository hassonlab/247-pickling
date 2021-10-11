import argparse
import os
import glob
import pickle

import pandas as pd


def load_pickle(pickle_name):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(pickle_name, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum)

    return df


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = file_name + add_ext
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        default='bert-large-uncased-whole-word-masking')
    parser.add_argument('--embedding-type', type=str, default='glove')
    parser.add_argument('--context-length', type=int, default=0)
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-hidden-states',
                        action='store_true',
                        default=False)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--subject', type=str, default='625')
    parser.add_argument('--history', action='store_true', default=False)
    parser.add_argument('--conversation-id', type=int, default=0)
    parser.add_argument('--pkl-identifier', type=str, default=None)
    parser.add_argument('--project-id', type=str, default=None)

    # import sys
    # sys.argv = ['tfsemb_concat.py', '--project-id', 'podcast', '--pkl-identifier', 'full', '--subject', '661', '--embedding-type', 'gpt2-xl', '--history', '--context-length', '1024']

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.subject == '625':
        num_convs = 54
    elif args.subject == '676':
        num_convs = 79
    else:
        num_convs = 1

    stra = args.embedding_type
    if 'gpt2' in args.embedding_type:
        stra = f'{stra}_cnxt_{args.context_length}'
    args.output_dir = os.path.join(os.getcwd(), 'results', args.project_id,
                                   args.subject, 'embeddings', stra,
                                   args.pkl_identifier)

    layer_folders = sorted(os.listdir(args.output_dir))

    for layer_folder in layer_folders:
        print(f'Merginng {layer_folder}')
        conversation_pickles = sorted(glob.glob(os.path.join(args.output_dir, layer_folder, '*')))
        n = len(conversation_pickles)
        if n != num_convs:
            print(f'Bad conversation size: found {n} out of {num_convs}',
                  f'in {args.output_dir}')
            continue
        
        all_df = []
        for conversation in conversation_pickles:
            conv_pkl = os.path.join(args.output_dir, conversation)
            all_df.append(load_pickle(conv_pkl))

        args.emb_out_dir = os.path.join(os.getcwd(), 'results', args.project_id,
                                        args.subject, 'pickles')
        strb = '_'.join([stra, layer_folder])
        args.emb_out_file = '_'.join(
            [args.subject, args.pkl_identifier, strb, 'embeddings'])

        all_df = pd.concat(all_df, ignore_index=True)

        all_df = all_df.to_dict('records')
        save_pickle(all_df, os.path.join(args.emb_out_dir, args.emb_out_file))


if __name__ == '__main__':
    main()
