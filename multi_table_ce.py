import os
import time
import torch
import pickle
import random
from tqdm import tqdm
from get_path import *
from my_model.BN import BN
from basic_info import BasicInfo
from query_parse import parse_sql,Attr
from inference_func import get_select_domain,get_join_condition_domain,sample_attr_value,get_attr_to_same,load_attr_dict_or_decoder

sample_times=1
sample_times2=1
try_lines=3
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
            with torch.no_grad():
                ce_res,time=one_sql(data_path,dataset,sql,verbose)
            # try:
            #     with torch.no_grad():
            #         ce_res,time=one_sql(data_path,dataset,sql,verbose)
            # except Exception:
            #     print(f"Error in {row_num} {sql}")
            #     with open(f'sqls/error_{dataset}_row_num.txt','a') as f:
            #         f.write(f'{row_num} || {sql}\n')
            #     with open(f'sqls/error_{dataset}.sql','a') as f:
            #         f.write(sql+'\n')
            #     continue
            with open('inference_time.csv','a') as f:
                f.write(f'{row_num},{time},{ce_res}\n')
            if no_try_line:
                continue
            if row_num==try_lines:
                return

def one_sql(data_path,dataset,sql,verbose=False):
    # 多表基数估计
    # 解析sql获取得到查询条件及连接条件
    if verbose:
        print(">> 解析SQL语句......")
    table_alias,select_conditions,join_conditions=parse_sql(sql)
    for sc in select_conditions:
        if(isinstance(sc,str)):
            continue
        sc[0].table=table_alias[sc[0].table]
    for jc in join_conditions:
        if(isinstance(jc,str)):
            continue
        jc[0].table=table_alias[jc[0].table]
        jc[2].table=table_alias[jc[2].table]
    if verbose:
        print(table_alias)
        print(select_conditions)
        print(join_conditions)
    # join_attr_pairs=[]
    # for jc in join_conditions:
    #     if(isinstance(jc,str)):
    #         continue
    #     apair=AttributePair(jc[0].attr,jc[2].attr)
    #     join_attr_pairs.append(apair)
    if verbose:
        # print(f'连接属性对: {join_attr_pairs}')
        # 从table_alias中获取sql涉及的表
        print(f">> 查询涉及的表: {table_alias.values()}")
    # 加载每一个表对应的BN到字典中
    if verbose:
        print(f">> 加载模型......")
    model_dicts=dict()
    model_info=dict()
    has_joined=dict()
    for table in table_alias.values():
        table_file=os.path.join(data_path,stats,f'{table}.csv')
        bn:BN=torch.load(get_bn_save_path(table_file), weights_only=False)
        with open(get_basic_info_save_path(table_file),'rb') as f:
            basic_info:BasicInfo=pickle.load(f)
        model_info[table]=basic_info
        model_dicts[table]=bn
        has_joined[table]=False
        # print(f'>> 表{table}有{basic_info.data_lens}行！')
    if verbose:
        print(">> 加载完成!")
    # 确定每个表涉及的连接属性及取值范围（此处取值范围为分通值）
    table_join_attrs,attrs_domain_sets=get_join_condition_domain(data_path,dataset,join_conditions)
    attrs_to_same,all_sampe_attrs,tuple_attrs=get_attr_to_same(join_conditions)
    if verbose:
        print('>> 连接条件相关信息')
        print(table_join_attrs) # 每个表涉及的连接属性
        print(join_conditions)
        # print(attrs_domain_sets) # 输出连接属性值域的关键词：最基本的等值连接属性
    # for k,v in table_join_attrs.items():
    #     res=load_attr_dict_or_decoder(data_path,dataset,table,attr)
    #     if res[0]:
    #         return False
    #     else:
    #         attr_decoder,bin_value_dict,unique_values=res[1:]
    #         original_value=np.array([v]).reshape(-1,1)
    #         value=attr_decoder.transform(original_value)
    #         transv=value.astype(int)[0]
    #     this_known_join_value.append((attr,int(transv[0])))
    # 确定每个表涉及的查询属性及取值范围（此处取值范围为桶值） 
    table_select_attrs,select_domain_sets=get_select_domain(data_path,dataset,select_conditions)
    if verbose:
        print('>> 查询条件相关信息')
        print(table_select_attrs)
        print(select_conditions)
        print(select_domain_sets)
    start=time.perf_counter()
    final_res=0
    if verbose:
        iteror=tqdm(range(sample_times))
    else:
        iteror=range(sample_times)
    for times in iteror:
    # 依次处理每个连接条件
        known_join_value=dict() # 已经采样出来的连接属性值
        res=1
        for jc in join_conditions:
            if isinstance(jc,str): # 排除掉AND
                continue
            # if verbose:
            #     print(f"处理连接条件：{jc}")
            l_attr=jc[0]
            r_attr=jc[2]
            if has_joined[l_attr.table]:
                if has_joined[r_attr.table]:# 两表均已连接
                    pass
                else: # 左表已连接
                    r_model=model_dicts[r_attr.table]
                    r_info=model_info[r_attr.table]
                    sump=progressive_sampling(r_model,r_info,known_join_value,r_attr.table,table_join_attrs,table_select_attrs,select_domain_sets,attrs_domain_sets,attrs_to_same)
                    r_lens=r_info.data_lens*sump
                    res=res*r_lens
            else:
                if has_joined[r_attr.table]: # 右表已连接
                    l_model=model_dicts[l_attr.table]
                    l_info=model_info[l_attr.table]
                    sump=progressive_sampling(l_model,l_info,known_join_value,l_attr.table,table_join_attrs,table_select_attrs,select_domain_sets,attrs_domain_sets,attrs_to_same)
                    l_lens=l_info.data_lens*sump
                    res=res*l_lens
                else: # 两表均未连接
                    # 先处理左表，从左表中渐进式采样获取对于连接属性的采样值
                    # 之后进行桶内采样获取采样值，再将采样值传入下一个表的分桶器，获取对应取值
                    # 统一保存的是原始的取值，而非分桶的取值
                    l_model=model_dicts[l_attr.table]
                    l_info=model_info[l_attr.table]
                    sump=progressive_sampling(l_model,l_info,known_join_value,l_attr.table,table_join_attrs,table_select_attrs,select_domain_sets,attrs_domain_sets,attrs_to_same)
                    l_lens=l_info.data_lens*sump
                    r_model=model_dicts[r_attr.table]
                    r_info=model_info[r_attr.table]
                    sump=progressive_sampling(r_model,r_info,known_join_value,r_attr.table,table_join_attrs,table_select_attrs,select_domain_sets,attrs_domain_sets,attrs_to_same)
                    r_lens=r_info.data_lens*sump
                    res=res*l_lens*r_lens
        final_res+=res/sample_times
    end=time.perf_counter()
    return final_res,end-start

def progressive_sampling(model,info:BasicInfo,known_join_value,table,table_join_attrs,table_select_attrs,select_domain_sets,attrs_domain_sets,attrs_to_same):
    '''
    获取在某个表上对查询条件和连接条件下的连接属性边缘概率分布P(A∈q(A),C∈J(C))
    并按照C的概率分布进行采样得到分桶采样值
    '''
    # attrs_domain_sets 是连接属性的可能取值范围
    this_table_select_attrs=table_select_attrs.get(table,set())
    this_table_join_attrs=table_join_attrs.get(table,set())
    this_known_join_value=[]
    for k,v in known_join_value.items():
        for attr in this_table_join_attrs:
            if Attr(table,attr) in k:
                this_known_join_value.append((attr,v))
    res_p=0
    for i in range(sample_times2):
        this_sample_value=this_known_join_value.copy()
        once_p=1
        for attr in info.attr_order:
            big_attr=Attr(table,attr)
            sump=1
            if attr in this_table_select_attrs:
                attr_value,sump=sample_attr_value(model,info.cols,select_domain_sets,big_attr,this_sample_value)
                if attr_value!=None:
                    this_sample_value.append((attr,attr_value))
            elif attr in this_table_join_attrs:
                attr_value,sump=sample_attr_value(model,info.cols,attrs_domain_sets,big_attr,this_sample_value)
                if attr_value!=None:
                    this_sample_value.append((attr,attr_value))
                    # 此处还应增加依桶内分布采样下一个属性值
                    same_attrs=attrs_to_same[big_attr]
                    if known_join_value.get(same_attrs,None)==None:
                        known_join_value[same_attrs]=attr_value
            else:
                continue
            once_p=once_p*sump
        res_p+=once_p
    return res_p/sample_times2

if __name__=='__main__':
    # const
    job='sqls/job_sub_query.sql'
    job_true='sqls/job_sub_query_truecard.txt'
    ceb='sqls/stats_sub_query.sql'
    ceb_true='sqls/stats_sub_query_truecard.txt'
    data_path='datasets'
    stats='stats'
    imdb='imdb' 

    print('>> 开始进行基数估计......')
    # main(data_path,stats,ceb,True)
    main(data_path,stats,ceb)
    # res,cost_time=one_sql(data_path,stats,"SELECT COUNT(*) FROM tags as t, posts as p WHERE p.Id = t.ExcerptPostId AND p.CreationDate>='2010-07-20 02:01:05'::timestamp;||16",verbose=True)
    # print(f'{res},{cost_time}')
    print('>> 完毕  AvA!')