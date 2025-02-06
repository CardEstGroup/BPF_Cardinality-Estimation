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
from bidict import bidict
from my_model.BN import BN
from inference_func import *
from basic_info import BasicInfo
# from memory_profiler import profile
from query_parse import parse_sql,Attr
# hyper parameters
device='cuda'
sample_times=1
sample_times2=1
try_lines=10
no_try_line=True

def main(data_path,dataset,query_file,verbose=False):
    with open('inference_time.csv','w') as f:
        f.write('sql_index,time,ce_res\n')
    with open(query_file,'r') as f:
        row_num=0
        if verbose:
            itertor=f
        else:
            itertor=tqdm(f)
        for line in itertor:
            row_num+=1
            sql=line.strip()
            if verbose:
                print(f">> 处理SQL: {sql}")
            # with torch.no_grad():
            #     ce_res,time=one_sql(data_path,dataset,sql,verbose)
            try:
                with torch.no_grad():
                    ce_res,time=one_sql(data_path,dataset,sql,verbose)
            except Exception:
                print(f"Error in {row_num} {sql}")
                with open(f'sqls/error_{dataset}_row_num.txt','a') as f:
                    f.write(f'{row_num} || {sql}\n')
                with open(f'sqls/error_{dataset}.sql','a') as f:
                    f.write(sql+'\n')
                continue
            with open('inference_time.csv','a') as f:
                f.write(f'{row_num},{time},{ce_res}\n')
            if no_try_line:
                continue
            if row_num==try_lines:
                return
class AttributePair:
    def __init__(self,key,value):
        self.key=key
        self.value=value
    def get_another(self, key):
        if key==self.key:
            return self.value
        elif key==self.value:
            return key

def one_sql(data_path,dataset,sql,verbose=False):
    # 多表基数估计
    # 解析sql获取得到查询条件及连接条件
    if verbose:
        print(">> 解析SQL语句......")
    # table_alias,select_conditions,join_conditions=parse_sql(sql)
    table_alias,select_conditions,join_conditions=parse_sql(sql,tolower=True)
    for sc in select_conditions:
        if(isinstance(sc,str)):
            continue
        sc[0].table=table_alias[sc[0].table]
    if verbose:
        print(table_alias)
        print(select_conditions)
    join_attr_pairs=[]
    # 加载每一个表对应的BN到字典中
    if verbose:
        print(f">> 加载模型......")
    model_dicts=dict()
    model_info=dict()
    for table in table_alias.values():
        table_file=os.path.join(data_path,dataset,f'{table}.csv')
        bn:BN=torch.load(get_bn_save_path(table_file), weights_only=False)
        with open(get_basic_info_save_path(table_file),'rb') as f:
            basic_info:BasicInfo=pickle.load(f)
        model_info[table]=basic_info
        model_dicts[table]=bn
    if verbose:
        print(">> 加载完成!")
    # 确定每个表涉及的连接属性及可能取值
    table_select_attrs,select_domain_sets=get_select_domain(data_path,dataset,select_conditions)
    if verbose:
        print(table_select_attrs)
        print(select_conditions)
        print(select_domain_sets)
    # 多次采样取平均
    start=time.perf_counter()
    all_res=0
    # sample_times=1
    if verbose:
        print('>> 基数估计中......')
    # if verbose:
        # itertor=tqdm(range(sample_times))
    # else:
        # itertor=range(sample_times)
    # for i in itertor:
    table=list(model_dicts.keys())[0]
    model=model_dicts[table]
    info=model_info[table]
    start=time.perf_counter()
    res_p=progressive_sampling(model,info,table,table_select_attrs,select_domain_sets)
    all_res=res_p*info.data_lens
    end=time.perf_counter()
    if verbose:
        print(f'查询语句{sql}\n基数估计值: {all_res},推理时间: {end-start}')
    return all_res,end-start

# 采样获取到的应为已知部分给定连接属性取值和其余连接属性可能取值，加上查询条件后的渐进式采样的一次值
def progressive_sampling(model,info:BasicInfo,table,table_select_attrs,select_domain_sets):
    # 合并所有的涉及的属性
    this_attrs=table_select_attrs.get(table,set())
    # 合并已知的连接属性的值到know_value中
    this_known_value=[]
    # 修改连接属性的domain_sets并和查询属性的domain_sets进行合并
    this_domain_sets=dict()
    for k,v in select_domain_sets.items():
        this_domain_sets[k]=v
    # 进行渐进式采样得到结果
    res_p=0
    for i in range(sample_times2):
    # for i in tqdm(range(sample_times2)):
        new_sample_value=[]
        once_p=1
        for attr in info.attr_order:
            if attr in this_attrs:
                attr_value,sump=sample_attr_value(model,info.cols,this_domain_sets,Attr(table,attr),this_known_value)
                if attr_value!=None:
                    new_sample_value.append((attr,attr_value))
                once_p=once_p*sump
        res_p+=once_p
    res_p/=sample_times2
    return res_p
def get_bidict_another(bi_dict,value):
    if value in bi_dict.keys():
        return bi_dict[value]
    else:
        return bi_dict.inverse[value]
def add_bidict_value(bi_dict,key,value):
    keys1=bi_dict.keys()
    keys2=bi_dict.inverse.keys()
    if (key not in keys1) and (value not in keys1):
        if (key not in keys2) and (value not in keys2):
            bi_dict[key]=value
            return True
    return False
def add_know_join_value(known_join_value,known_join_value_attr,join_value,join_attr_pair):
    if join_value[0] not in known_join_value_attr:
        known_join_value.append(join_value)
    if join_attr_pair.get_another(join_value[0]) not in known_join_value_attr:
        known_join_value.append((join_attr_pair.get_another(join_value[0]),join_value[1]))
    
if __name__=='__main__':
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
    # 估计值 570926208.0
    test_sql2=r"SELECT COUNT(*) FROM badges as b, comments as c, users as u WHERE c.UserId = u.Id AND b.UserId = u.Id AND b.Date<='2014-09-11 14:33:06'::timestamp AND c.Score=0;"
    true_card2=10220614
    # 估计值 64139702272.0
    test_sql3=r"SELECT COUNT(*) FROM postHistory as ph, comments as c, users as u WHERE c.UserId = u.Id AND ph.UserId = u.Id AND ph.PostHistoryTypeId=1 AND c.Score=0;"
    true_card3=1458075
    test_sql4=r"SELECT COUNT(*) FROM postLinks as pl, comments as c, posts as p, users as u WHERE c.UserId = u.Id AND p.Id = pl.PostId AND p.OwnerUserId = u.Id AND p.CommentCount<=18 AND p.CreationDate>='2010-07-23 07:27:31'::timestamp AND p.CreationDate<='2014-09-09 01:43:00'::timestamp;"
    true_card4=699302
    # 估计值 79761.3515625
    error_sql=r"SELECT COUNT(*) FROM users as u, comments as c WHERE c.UserId = u.Id AND u.DownVotes<=0 AND u.UpVotes>=0 AND u.CreationDate>='2010-08-23 16:21:10'::timestamp AND u.CreationDate<='2014-09-02 09:50:06'::timestamp AND c.Score=0;"
    error_sql=r"SELECT COUNT(*) FROM postHistory as ph, comments as c, users as u WHERE c.UserId = u.Id AND ph.UserId = u.Id AND ph.PostHistoryTypeId=1 AND ph.CreationDate>='2010-09-14 11:59:07'::timestamp;"
    error_sql=r"SELECT COUNT(*) FROM comments as c WHERE c.CreationDate>='2010-07-26 20:21:15'::timestamp AND c.CreationDate<='2014-09-13 18:12:10'::timestamp;"
    print('>> 开始进行基数估计......')
    # with torch.no_grad():
        # 真实值：134887
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM badges as b WHERE b.Date<='2014-09-11 14:33:06'::timestamp;",verbose=True)
        # 1 79633
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM users as u WHERE u.DownVotes>=0 AND u.DownVotes<=0;",verbose=True)
        # 22 39578
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.CommentCount>=0 AND p.CommentCount<=25;",verbose=True)
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM comments as c WHERE c.Score=0;",verbose=True)
        # 134887
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM votes as v WHERE v.BountyAmount<=100;",verbose=True)
        # 13 1652
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.AnswerCount<=4 AND p.CommentCount>=0 AND p.CommentCount<=17;",verbose=True)
        # 5 42172
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.AnswerCount>=0 AND p.CommentCount>=0;",verbose=True)
        # 42921
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.Score>=-1 AND p.Score<=14;||89806||1",verbose=True)
        # one_sql('test_data','',"SELECT COUNT(*) FROM posts as p WHERE p.Score>=-1 AND p.Score<=14;||89806||1",verbose=True)
        # one_sql('test_data','',"SELECT COUNT(*) FROM votes as v WHERE v.BountyAmount<=100;||13||1652",verbose=True)
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.PostTypeId=1 AND p.Score<=35 AND p.AnswerCount=1 AND p.CommentCount<=17 AND p.FavoriteCount>=0;||5209||1",verbose=True)
        # one_sql(data_path,stats,"SELECT COUNT(*) FROM posts as p WHERE p.PostTypeId=1 AND p.ViewCount>=0 AND p.ViewCount<=4157 AND p.FavoriteCount=0 AND p.CreationDate<='2014-09-08 09:58:16'::timestamp;||879||1",verbose=True)
    # one_sql(data_path,stats,error_sql,verbose=True)
    # main(data_path,stats,'sqls/error_stats.sql')
    # main(data_path,stats,ceb,True)
    main(data_path,stats,'sqls/stats_single_query.sql')
    # main(data_path,imdb,'sqls/job_single_query.sql')
    # with torch.no_grad():
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_Id=112;||2",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_Id=113;||3",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_Id=117;||2",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_Id=101;||16",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_Id=8200;||3",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM cast_info ci WHERE ci.role_Id=2;||11",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM cast_info ci WHERE ci.role_Id=4;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM cast_info ci WHERE ci.role_Id=7;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_Id=398;||6",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_Id=3;||7",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_Id=105;||2",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM title t WHERE t.kind_Id=1;||2",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_Id=16;||12",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM title t WHERE t.production_year>2010 AND t.kind_Id=1;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info mi WHERE mi.info_type_Id=8;||7",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_info_idx mi_idx WHERE mi_idx.info_type_Id=100;||5",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM title t WHERE t.production_year>1950 AND t.kind_Id=1;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM title t WHERE t.production_year>2000 AND t.kind_Id=1;||2",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_companies mc WHERE mc.company_Id=22956;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM title t WHERE t.production_year>2005 AND t.kind_Id=1;||1",verbose=True)
    #     one_sql(data_path,imdb,"SELECT COUNT(*) FROM movie_keyword mk WHERE mk.keyword_Id=7084;||2",verbose=True)
    print('>> 完毕  AvA!')
    # print("True Card: 79851,10220614")
    