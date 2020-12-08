python -u src/preprocess.py 1>preprocess.log 2>&1 
python -u src/get_stat_feat.py 1>stat.log 2>&1
python -u src/get_ctr_feat.py 1>ctr.log 2>&1
python -u src/get_prevday_feat.py 1>prevday.log 2>&1
python -u src/get_nowaday_feat.py 1>nowaday.log 2>&1
python -u src/get_w2v_feat.py 1>w2v.log 2>&1
python -u src/concat_feat.py 1>concat.log 2>&1
python -u src/lgb.py 1>lgb.log 2>&1