import gc
import time
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import MinMaxScaler
import warnings
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
warnings.filterwarnings('ignore')

df = pd.read_pickle('./data/df.pkl')
to_del_cols = df.columns

def prevday(df, f):
     # 前一日点击次数, 曝光次数
    tmp = df.groupby(f+['pt_d'], as_index=False).label.agg({
        '{}_prevday_click_count'.format('_'.join(f)): 'sum',
        '{}_prevday_count'.format('_'.join(f)): 'count',
        '{}_prevday_ctr'.format('_'.join(f)): 'mean',
    })
    tmp['pt_d'] += 1
    # 第七天视作第一天的前一天
    tmp0 = tmp[tmp.pt_d==8].copy()
    tmp0['pt_d'] = 1
    # 第二天视作第十天的前一天
    tmp1 = tmp[tmp.pt_d==3].copy()
    tmp1['pt_d'] = 10
    
    tmp = pd.concat([tmp, tmp0, tmp1])
    df = df.merge(tmp, on=f+['pt_d'], how='left')
    return df

for f in tqdm([
    ['uid'],
    ['adv_id'],
    ['adv_id', 'slot_id'],
    ['adv_id', 'net_type'],
    ['adv_id', 'spread_app_id'],
    ['adv_id', 'device_name'],
    ['task_id'],
    ['task_id', 'slot_id'],
    ['task_id', 'net_type'],
    ['task_id', 'spread_app_id'],
    ['task_id', 'device_name'],
    ['slot_id'],
    ['net_type'],
    ['spread_app_id'],
    ['device_name'],
]):
    df = prevday(df, f)

df = pd.concat([df[df.label!=0], df[df.label==0].sample(frac=0.5, random_state=2020)], ignore_index=True)

for col in to_del_cols:
    del df[col]

df = reduce_mem(df)
df.to_pickle('./data/df_prevday.pkl')