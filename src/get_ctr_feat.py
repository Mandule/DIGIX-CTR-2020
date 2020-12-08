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

def ctr(train_df, test_df, f):
    train_df['{}_ctr'.format('_'.join(f))] = 0.0
    test_df['{}_ctr'.format('_'.join(f))] = 0.0
    
    trn_x = train_df[f+['label']].reset_index(drop=True)
    test_x = test_df[f]
    enc_df = trn_x.groupby(f, as_index=False)['label'].agg({'{}_ctr'.format('_'.join(f)) : 'mean'})
    test_x = test_x.merge(enc_df, on=f, how='left')
    test_x['{}_ctr'.format('_'.join(f))] = test_x['{}_ctr'.format('_'.join(f))]
    test_df['{}_ctr'.format('_'.join(f))] += test_x['{}_ctr'.format('_'.join(f))]
    
    for i in [1,2,3,4,5,6,7]:
        trn_x = train_df[train_df.pt_d!=i][f+['label']].reset_index(drop=True)
        val_x = train_df[train_df.pt_d==i][f].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['label'].agg({'{}_ctr'.format('_'.join(f)) : 'mean'})
        val_x = val_x.merge(enc_df, on=f, how='left')
        val_x['{}_ctr'.format('_'.join(f))] = val_x['{}_ctr'.format('_'.join(f))]
        train_df.loc[train_df.pt_d==i, '{}_ctr'.format('_'.join(f))] = val_x['{}_ctr'.format('_'.join(f))].values
    
    return train_df, test_df

train_df = df[df.label!=-1].reset_index(drop=True)
test_df = df[df.label==-1].reset_index(drop=True)

feats = ['uid', 'age', 'gender', 'city', 'city_rank', 'career',  'residence',
         'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'online_start', 'online_end','communication_avgonline_30d',
         'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'indu_name',
         'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'his_app_size', 'his_on_shelf_time',
         'device_name', 'device_size', 'emui_dev', 'list_time', 'device_price',
         'slot_id', 'net_type']

for f in tqdm([[f] for f in feats] + [
    ['adv_id', 'city'],
    ['adv_id', 'city_rank'],
    ['adv_id', 'residence'],
    ['adv_id', 'age'],
    ['adv_id', 'gender'],
    ['adv_id', 'career'],
    ['adv_id', 'up_membership_grade'],
    ['adv_id', 'up_life_duration'],
    ['adv_id', 'membership_life_duration'],
    ['adv_id', 'consume_purchase'],
    ['adv_id', 'spread_app_id'],
    ['adv_id', 'device_name'],
    ['adv_id', 'emui_dev'],
    ['adv_id', 'device_price'],
    ['adv_id', 'slot_id'],
    ['adv_id', 'net_type'],
    ['task_id', 'city'],
    ['task_id', 'city_rank'],
    ['task_id', 'residence'],
    ['task_id', 'age'],
    ['task_id', 'gender'],
    ['task_id', 'career'],
    ['task_id', 'up_membership_grade'],
    ['task_id', 'up_life_duration'],
    ['task_id', 'membership_life_duration'],
    ['task_id', 'consume_purchase'],
    ['task_id', 'spread_app_id'],
    ['task_id', 'device_name'],
    ['task_id', 'emui_dev'],
    ['task_id', 'device_price'],
    ['task_id', 'slot_id'],
    ['task_id', 'net_type'],
    ['slot_id', 'net_type'],
    ['spread_app_id', 'slot_id'],
    ['spread_app_id', 'net_type'],
]):
    ctr(train_df, test_df, f)

df = pd.concat([train_df, test_df], ignore_index=True)
df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)

for col in to_del_cols:
    del df[col]

df = reduce_mem(df)
df.to_pickle('./data/df_ctr.pkl')