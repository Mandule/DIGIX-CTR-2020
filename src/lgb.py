import gc
import os
import time
from utils import *
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

df_train = pd.read_pickle('./data/df_train_new.pkl')
df_test = pd.read_pickle('./data/df_test_new.pkl')

feats = df_train.columns.drop(['label', 'pt_d', 'id']).values
df_trn = df_train[df_train.pt_d.isin([1,2,3,4])].reset_index(drop=True)
df_val = df_train[df_train.pt_d==7].reset_index(drop=True)
X_trn, Y_trn = df_trn[feats], df_trn.label
X_val, Y_val = df_val[feats], df_val.label

clf = LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    n_estimators=10000,
    num_leaves=63,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=2020,
    n_jobs=32,
)

clf.fit(
    X_trn, Y_trn,
    eval_set=[(X_val, Y_val)],
    eval_metric='auc',
    early_stopping_rounds=200,
    verbose=200,
)

best_iter = clf.best_iteration_
X_trn, Y_trn = df_train[feats], df_train.label
X_sub = df_test[feats]

clf = LGBMClassifier(
    objective='binary',
    learning_rate=0.05,
    n_estimators=best_iter,
    num_leaves=63,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=2020,
    n_jobs=32,
)

clf.fit(X_trn, Y_trn, verbose=200)
sub = clf.predict_proba(X_sub)[:, 1]
imp = clf.feature_importances_

pd.DataFrame({
    'id': df_test.id,
    'probability': sub.astype('float32'),
}).to_csv('submission_1.csv', index=None)
pd.DataFrame({
    'feat': feats,
    'imp': imp,
}).to_csv('feat_imp_1.csv', index=None)