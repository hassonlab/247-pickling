import os
from shutil import copy2

if __name__ == '__main__':
    src_subj_id = '661'
    full_emb = '_full_gpt2-xl_cnxt_1024_embeddings.pkl'
    trim_emb = '_trimmed_gpt2-xl_cnxt_1024_embeddings.pkl'

    prjct_dir = '/scratch/gpfs/hgazula/247-pickling/results/podcast/'
    src_file_path = os.path.join(prjct_dir, src_subj_id, 'pickles')

    full_src1 = os.path.join(src_file_path, src_subj_id + full_emb)
    trim_src1 = os.path.join(src_file_path, src_subj_id + trim_emb)

    for subject in [662, 717, 723, 741, 742, 743, 763, 798, 777]:
        full_dest = os.path.join(prjct_dir, str(subject), 'pickles',
                                 str(subject) + full_emb)
        trim_dest = os.path.join(prjct_dir, str(subject), 'pickles',
                                 str(subject) + trim_emb)
        copy2(full_src1, full_dest)
        try:
            copy2(trim_src1, trim_dest)
        except FileNotFoundError:
            pass
