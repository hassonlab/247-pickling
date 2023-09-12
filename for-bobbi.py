import glob
import os
import pickle
import pandas as pd
from pprint import pprint

if __name__ == "__main__":
    datum_files = glob.glob(
        "/projects/HASSON/247/data/conversations-car/798/*/misc/*datum_trimmed.txt"
    )
    # file_name = (
    #     "/scratch/gpfs/kw1166/247-pickling/results/tfs/7170/pickles/7_full_labels.pkl"
    # )
    # with open(file_name, "rb") as f:
    #     datum = pickle.load(f)
    # df = pd.DataFrame(datum["labels"])

    # for test_word in df.speaker.unique():
    #     if "Speaker" in word:
    #         continue

    for datum in datum_files:
        with open(datum, "r") as fn:
            datum_contents = [line.rstrip().split() for line in fn]

        flag = 0
        for word, *rest in datum_contents:
            if "inaudible" in word:
                flag = 1
                print(" ".join([word, *rest]))
        if flag:
            print(os.path.basename(datum))
            print("=" * 50)
