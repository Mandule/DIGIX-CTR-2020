import gc
import time
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
import warnings
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

def count(df, f):
    df['{}_count'.format(f)] = 0
    for i in [1,2,3,4,5,6,7]:
        map_dict = df[~df.pt_d.isin([i,10])][f].value_counts()
        df.loc[df.pt_d==i, '{}_count'.format(f)] = df[df.pt_d==i][f].map(map_dict).values
    map_dict = df[df.pt_d!=10][f].value_counts()
    df.loc[df.pt_d==10, '{}_count'.format(f)] = df[df.pt_d==10][f].map(map_dict).values
    return df

def cross(df, f1, f2):
    df['{}_{}_count'.format(f1, f2)] = 0
    df['{}_in_{}_ratio'.format(f2, f1)] = 0
    for i in [1,2,3,4,5,6,7]:
        tmp = df[~df.pt_d.isin([i,10])][[f1,f2]].groupby([f1,f2], as_index=False)[f1].agg({'{}_{}_count'.format(f1, f2): 'count'})
        df.loc[df.pt_d==i, '{}_{}_count'.format(f1, f2)] = df[df.pt_d==i][[f1,f2]].merge(tmp, on=[f1,f2], how='left')['{}_{}_count'.format(f1, f2)].values
    tmp = df[df.pt_d!=10][[f1,f2]].groupby([f1,f2], as_index=False)[f1].agg({'{}_{}_count'.format(f1, f2): 'count'})
    df.loc[df.pt_d==10, '{}_{}_count'.format(f1, f2)] = df[df.pt_d==10][[f1,f2]].merge(tmp, on=[f1,f2], how='left')['{}_{}_count'.format(f1, f2)].values
    df['{}_in_{}_ratio'.format(f2, f1)] = df['{}_{}_count'.format(f1,f2)] / df['{}_count'.format(f1)]
    return df

df = pd.read_pickle('./data/df.pkl')
to_del_cols = df.columns

# all count
for f in tqdm(['uid', 'age', 'gender', 'city', 'city_rank', 'career',  'residence',
         'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'online_start', 'online_end',
         'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'indu_name',
         'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'his_app_size', 'his_on_shelf_time',
         'device_name', 'device_size', 'emui_dev', 'list_time', 'device_price',
         'slot_id', 'net_type']):
    df = count(df, f)

# task_id, adv_id, adv_prim_id
for f in tqdm(['age', 'gender', 'city', 'city_rank', 'career',  'residence','up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'online_start', 'online_end']):
    df = cross(df, 'task_id', f)
    df = cross(df, 'adv_id', f)
    df = cross(df, 'adv_prim_id', f)

# slot_id, net_type
df = cross(df, 'slot_id', 'net_type')
df = cross(df, 'net_type', 'slot_id')

df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)
for col in to_del_cols:
    del df[col]

df = reduce_mem(df)
df.to_pickle('./data/df_stat.pkl')