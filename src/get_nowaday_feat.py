import gc
import time
import numpy as np
import pandas as pd
from utils import *
import warnings
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

df = pd.read_pickle('./data/df.pkl')
to_del_cols = df.columns

def nowaday_count(df, f):
    # count
    df['{}_nowaday_count'.format(f)] = df.groupby([f, 'pt_d']).label.transform('count')
    return df

def nowaday_cross(df, f1, f2):
    # count & ratio
    df['{}_{}_nowaday_count'.format(f1, f2)] = df.groupby([f1, f2, 'pt_d']).label.transform('count')
    df['{}_in_{}_nowaday_ratio'.format(f2, f1)] = df['{}_{}_nowaday_count'.format(f1, f2)] / df['{}_nowaday_count'.format(f1)]
    return df

for f in tqdm(['uid', 'adv_id', 'task_id', 'slot_id', 'net_type', 'spread_app_id']):
    df = nowaday_count(df, f)

for f1, f2 in tqdm([
    ['adv_id', 'slot_id'],
    ['adv_id', 'net_type'],
    ['adv_id', 'spread_app_id'],
    ['task_id', 'slot_id'],
    ['task_id', 'net_type'],
    ['task_id', 'spread_app_id'],
]):
    df = nowaday_cross(df, f1, f2)

df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)
for f in tqdm(df.columns):
    if 'count' in f:
        df.loc[df.label==-1, f] *= int(0.8 * df[df.label!=-1][f].max() / df[df.label==-1][f].max())

for col in to_del_cols:
    del df[col]

df = reduce_mem(df)
df.to_pickle('./data/df_nowaday.pkl')