import argparse
import os
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.subject == '625':
        num_convs = 54
    elif args.subject == '676':
        num_convs = 79
    else:
        num_convs = 1

    stra = '_'.join([args.embedding_type, 'cnxt', str(args.context_length)])
    args.output_dir = os.path.join(os.getcwd(), 'results', args.subject,
                                   'embeddings', stra)

    conversation_pickles = sorted(os.listdir(args.output_dir))
    assert len(conversation_pickles) == num_convs, 'Bad conversation size'

    all_df = []
    for conversation in conversation_pickles:
        conv_pkl = os.path.join(args.output_dir, conversation)
        all_df.append(load_pickle(conv_pkl))

    emb_out_dir = os.path.join(os.getcwd(), 'results', args.subject, 'pickles')
    emb_out_file = '_'.join([args.subject, stra, 'embeddings'])

    all_df = pd.concat(all_df, ignore_index=True)
    all_df = all_df.to_dict('records')
    save_pickle(all_df, os.path.join(emb_out_dir, emb_out_file))


if __name__ == '__main__':
    main()
