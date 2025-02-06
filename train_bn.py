# %%
# imports
import os
import time
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from get_path import *
from basic_info import BasicInfo
from query_parse import parse_sql
# hyper parameters
device='cuda'
sample_times=10
# const
job='sqls/job_sub_query.sql'
job_true='sqls/job_sub_query_truecard.txt'
ceb='sqls/stats_sub_query.sql'
ceb_true='sqls/stats_sub_query_truecard.txt'
data_path='datasets'
stats='stats'
imdb='imdb'
test_sql1=r"SELECT COUNT(*) FROM users as u, badges as b WHERE b.UserId= u.Id AND u.UpVotes>=0;"
true_card1=79851
test_sql2=r"SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp AND c.Score=0;"
true_card2=10220614

# %%
# basic init
query_file=job
query_file=ceb
dataset=imdb
dataset=stats
sql=test_sql1
# %%
# 训练BN
from train_func import train_model,count_attr_fre
print(f'>> 获取需要训练的表......')
sort_fre=count_attr_fre(query_file)
need_tables=set()
for sf in sort_fre:
    need_tables.add(sf[0].split('.')[0])
# print(f'>> 需要训练的表: {need_tables}')
# need_tables=['badges','postLinks','users']
# need_tables=['cast_info', 'movie_info_idx', 'movie_companies', 'movie_info', 'movie_keyword']
print(f'>> 需要训练的表: {need_tables}')
with open('train_time.csv','w') as f:
    f.write('table_name,need_time\n')
for i,table in enumerate(need_tables):
    # 训练一个对应的BN
    table_file=os.path.join(data_path,dataset,f'{table}.csv')
    # if(os.path.exists(get_bn_save_path(table_file))):
        # continue
    start=time.perf_counter()
    success=train_model(query_file,table_file)
    # success=True
    # print(i**2)
    end=time.perf_counter()
    with open('train_time.csv','a') as f:
        f.write(f'{table},{end-start}\n')
    if success:
        if(len(need_tables)-i==1):
            print(f'>> [{i+1}/{len(need_tables)}] Finish training on {table}, don\'t have any models to train!')
        else:
            print(f'>> [{i+1}/{len(need_tables)}] Finish training on {table}, still have {len(need_tables)-1-i} models to train!')
    else:
        print(f'Something went wrong during train model from {table}!')
        break
print('All training finished, congritulations!!!')
print('AvA')
