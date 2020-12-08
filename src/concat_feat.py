import gc
import pandas as pd
from utils import *
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

print('reading data')
df = pd.read_pickle('data/df.pkl')
df_ctr = pd.read_pickle('data/df_ctr.pkl')
df_stat = pd.read_pickle('data/df_stat.pkl')
# df_nowaday = pd.read_pickle('data/df_nowaday.pkl')
# df_w2v = pd.read_pickle('data/df_w2v.pkl')

print('labelencoding')
df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)

for feat in tqdm(df.columns.drop(['label', 'id', 'pt_d'])):
    df[feat] = LabelEncoder().fit_transform(df[feat])
df = reduce_mem(df)
gc.collect()

print('saving data')
df_feats = pd.concat([df, df_ctr, df_stat], axis=1)

df_train = df_feats[df_feats.label!=-1].reset_index(drop=True)
df_train.to_pickle('data/df_train_new.pkl')
del df_train
gc.collect()

df_test = df_feats[df_feats.label==-1].reset_index(drop=True)
df_test.to_pickle('data/df_test_new.pkl')