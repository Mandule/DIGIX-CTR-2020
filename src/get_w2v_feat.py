import gc
import time
import numpy as np
import pandas as pd
from utils import *
from gensim.models import Word2Vec
import warnings
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

df = pd.read_pickle('./data/df.pkl')
to_del_cols = df.columns

def emb(df, f1, f2, emb_size):
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].apply(lambda s: [str(w) for w in s]).values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, workers=32, sg=1, seed=2020)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix, dtype='float16')
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    
    del model, emb_matrix, sentences
    df = df.merge(tmp, on=f1, how='left')
    del tmp
    return df

for f1, f2, emb_size in tqdm([
    ['uid', 'adv_id', 16],
    ['uid', 'task_id', 16],
    ['uid', 'adv_prim_id', 8],
    ['uid', 'spread_app_id', 8],
    ['uid', 'tags', 8],
    ['uid', 'slot_id', 4],
    ['uid', 'net_type', 4],
    ['adv_id', 'emui_dev', 8],
    ['adv_id', 'spread_app_id', 8],
    ['adv_id', 'tags', 8],
    ['adv_id', 'device_name', 8],
    ['adv_id', 'residence', 8],
    ['adv_id', 'city', 8],
    ['adv_id', 'age', 4],
    ['adv_id', 'slot_id', 4],
    ['adv_id', 'net_type', 4],
    ['task_id', 'emui_dev', 8],
    ['task_id', 'spread_app_id', 8],
    ['task_id', 'tags', 8],
    ['task_id', 'device_name', 8],
    ['task_id', 'residence', 8],
    ['task_id', 'city', 8],
    ['task_id', 'age', 8],
    ['task_id', 'slot_id', 4],
    ['task_id', 'net_type', 4],
]):
    df = emb(df, f1, f2, emb_size)

df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)
for col in to_del_cols:
    del df[col]

df = reduce_mem(df)
df.to_pickle('./data/df_w2v.pkl')
