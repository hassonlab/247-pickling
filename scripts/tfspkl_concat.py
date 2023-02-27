import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


pca_to = 50

def run_pca(pca_to, df):
    pca = PCA(n_components=pca_to, svd_solver="auto", whiten=True)

    df_emb = df["embeddings"]
    embs = np.vstack(df_emb.values)

    pca_output = pca.fit_transform(embs)
    df["embeddings"] = pca_output.tolist()

    return df

# get base
whisper_base = pd.DataFrame(pd.read_pickle('/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/whisper-tiny.en-encoder-new/full/base_df.pkl'))
gpt2_base = pd.DataFrame(pd.read_pickle('/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/gpt2/full/base_df.pkl'))

# get embs
whisper_emb = pd.DataFrame(pd.read_pickle('/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/whisper-tiny.en-encoder-new/full/cnxt_0001/layer_04.pkl'))
gpt2_emb = pd.DataFrame(pd.read_pickle('/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/gpt2/full/cnxt_1024/layer_08.pkl'))

# remove nans from gpt2 base
gpt2_base = gpt2_base.dropna(subset=["onset","offset"])

# shift indeces for gpt2 - going from n-1 to n
gpt2_emb.embeddings = gpt2_emb.embeddings.shift(-1)

# align emb_gpt2
gpt2_emb = gpt2_emb[gpt2_emb.index.isin(gpt2_base.index)]
gpt2_emb.reset_index(drop=True, inplace=True)

assert(len(gpt2_emb.index) == len(whisper_emb.index))

# pca embeddings
whisper_emb = run_pca(pca_to,whisper_emb)
gpt2_emb = run_pca(pca_to, gpt2_emb)

# concat embs ...
concat_base = whisper_base
concat_emb = whisper_emb

for i in range(0,len(concat_emb.index)):
    concat_emb.embeddings.iloc[i] = concat_emb.embeddings.iloc[i] + gpt2_emb.embeddings.iloc[i]

# save
output_emb = '/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/whisper-en-gpt2-concat/full/cnxt_0001/layer_01.pkl'
output_base = '/scratch/gpfs/ln1144/247-pickling/results/podcast/777/pickles/embeddings/whisper-en-gpt2-concat/full/base_df.pkl'

concat_emb.to_pickle(output_emb)
concat_base.to_pickle(output_base)

