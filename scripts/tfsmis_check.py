import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SIDS = [625, 676, 798, 7170]

for sid in SIDS:

    # read base df
    df = pd.read_pickle(f'/scratch/gpfs/ln1144/247-pickling/results/tfs/{sid}/pickles/embeddings/whisper-tiny.en-decoder/full/base_df.pkl')

    df = pd.DataFrame(df)

    for conv_id in df.conversation_id.unique():

        conv_df = df[df['conversation_id' == conv_id]]

        for idx in range(len(conv_df)):

            diff = np.diff(conv_df.onset) < 0
                
        perc =  np.count_nonzero(diff==1) / len(diff)
        diff = df['diff'].tolist()

        fig = plt.figure()
        plt.plot(df.onset)
        t = str(sid) + '_' + str(conv_id)
        plt.title(t)
        st = str(perc) + ' percent of all words'
        plt.suptitle(st)

        fname = f'/scratch/gpfs/ln1144/247-pickling/results/{sid}_{conv_id}_onset_check_new.png'

        plt.savefig(fname)
        plt.close()
        


