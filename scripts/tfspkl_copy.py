import glob
import os
from shutil import copy2

PRJCT_DIR = "/scratch/gpfs/hgazula/247-pickling/results/podcast/"


def create_dest_filename(src_file, subject):
    # get file name
    file_name = os.path.split(src_file)[-1]

    # split at first underscore to replace subject ID
    # TODO: regex
    temp = file_name.split("_")
    temp = "_".join(temp[1:])

    dest_file_name = str(subject) + "_" + temp

    full_dest = os.path.join(PRJCT_DIR, str(subject), "pickles", dest_file_name)
    return full_dest


if __name__ == "__main__":
    src_subj_id = "661"
    full_emb = "_full_*_embeddings.pkl"
    trim_emb = "_trimmed_*_embeddings.pkl"

    src_file_path = os.path.join(PRJCT_DIR, src_subj_id, "pickles")

    full_src1 = sorted(glob.glob(os.path.join(src_file_path, src_subj_id + full_emb)))
    trim_src1 = sorted(glob.glob(os.path.join(src_file_path, src_subj_id + trim_emb)))

    for subject in [662, 717, 723, 741, 742, 743, 763, 798, 777]:
        for src_file in full_src1:
            full_dest = create_dest_filename(src_file, subject)
            try:
                copy2(src_file, full_dest)
            except FileNotFoundError:
                pass

        for trim_file in trim_src1:
            trim_dest = create_dest_filename(trim_file, subject)
            try:
                copy2(trim_src1, trim_dest)
            except FileNotFoundError:
                pass
