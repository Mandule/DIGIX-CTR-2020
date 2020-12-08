import gc
from utils import *
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

train_df = pd.read_pickle('./data/original/train.pkl')
test_df = pd.read_pickle('./data/original/test_B.pkl')

train_df['id'] = -1
test_df['label'] = -1
df = pd.concat([train_df, test_df], ignore_index=True)

del test_df
del train_df
gc.collect()

df['online_start'] = df.communication_onlinerate.progress_apply(lambda s: int(s.split('^')[0]))
df['online_end'] = df.communication_onlinerate.progress_apply(lambda s: int(s.split('^')[-1]))
del df['communication_onlinerate']
del df['app_score']
gc.collect()

df['device_size'] = pd.cut(df.device_size, bins=10, labels=False)
for feat in tqdm(df.columns.drop(['label', 'pt_d', 'id'])):
    df[feat] = LabelEncoder().fit_transform(df[feat])

df = reduce_mem(df)
df.to_pickle('./data/df.pkl')