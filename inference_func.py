# imports
import os
import gc
import torch
import pickle
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from get_path import *
from basic_info import BasicInfo
# from line_profiler import profile
from query_parse import parse_sql,Attr
# hyper parameters
device='cuda'
sample_times=10
bin_nums=1900

def domain2binvalue(data_path,dataset,domain,attr:Attr):
    return [slice(0,bin_nums)]
    res=load_attr_dict_or_decoder(data_path,dataset,attr.table,attr.attr)
    if res[0]:
        return False
    else:
        attr_decoder,bin_value_dict,unique_values=res[1:]
        original_value=np.array(list(domain)).reshape(-1,1)
        value=attr_decoder.transform(original_value)
        results=[]
        for bin_index in np.unique(value):
            results.append((int(bin_index),1.0))
        results.append(slice(0,0))
        return results
    res=load_attr_dict_or_decoder(data_path,dataset,attr.table,attr.attr)
    if res[0]:
        return False
    else:
        attr_decoder,bin_value_dict,unique_values=res[1:]
        original_value=np.array(list(domain)).reshape(-1,1)
        value=attr_decoder.transform(original_value)
        results=[]
        counts_dict=dict()
        total_dict=dict()
        for ov,v in zip(original_value,value):
            ov=ov[0]
            v=int(v[0])
            total=sum(bin_value_dict[v].values())
            total_dict[v]=total
            for k,counts in bin_value_dict[v].items():
                if k==ov:
                    counts_dict[v]=counts_dict.get(v,0)+counts
                    break
        for k,v in counts_dict.items():
            results.append((k,v/total_dict[k]))
        results.append(slice(0,0))
        # value=value_set2bin_prob(bin_value_dict,value)
        return results
def get_original_bin_value(data_path,dataset,base_domain,table,attr):
    res=load_attr_dict_or_decoder(data_path,dataset,table,attr)
    if res[0]:
        return False
    else:
        attr_decoder,bin_value_dict,unique_values=res[1:]
        original_value=np.array(base_domain).reshape(-1,1)
        value=attr_decoder.transform(original_value)
        results=[]
        value=value.astype(int)
        counts_dict=dict()
        total_dict=dict()
        for i,v in enumerate(value):
            av=v[0]
            ov=original_value[i][0]
            if total_dict.get(av,0)==0:
                total_num=sum(bin_value_dict[av].values())
                total_dict[av]=total_num
            counts_dict[av]=counts_dict.get(av,0)+bin_value_dict[av][ov]
            # results.append((av,bin_value_dict[av][ov]/total_num))
        for k,v in counts_dict.items():
            results.append((k,v/total_dict[k]))
        results.append(slice(0,0))
        return results
def get_attr_to_same(join_conditions):
    all_same_attrs=[] # 所有等值连接的相同属性
    for jc in join_conditions:
        if isinstance(jc,str):
            continue
        else:
            if len(all_same_attrs)==0: # 初始化等值连接的属性
                same_attr=set()
                same_attr.add(jc[0])
                same_attr.add(jc[2])
                all_same_attrs.append(same_attr)
                continue
            notin=True
            for same_attr in all_same_attrs:
                lin=jc[0] in same_attr
                rin=jc[2] in same_attr
                if lin and not rin:
                    same_attr.add(jc[2])
                    notin=False
                    break
                elif not lin and rin:
                    same_attr.add(jc[0])
                    notin=False
                    break
                elif lin and rin:
                    notin=False
            if notin: # 另一对等值连接的属性
                same_attr=set()
                same_attr.add(jc[0])
                same_attr.add(jc[2])
                all_same_attrs.append(same_attr)
    tuple_attrs=[]
    attrs_to_same=dict() # 获取连接属性到所有连接属性的映射
    for same_attr in all_same_attrs:
        al=list(same_attr)
        tal=tuple(al)
        tuple_attrs.append(tal)
        for k in tal:
            attrs_to_same[k]=tal
    return attrs_to_same,all_same_attrs,tuple_attrs
def get_join_original_domain(data_path,dataset,join_conditions):
    table_join_attrs=dict()
    attrs_trans_domain=dict()
    base_attr_domain_sets=dict()
    all_join_attrs=set()
    attrs_to_same,all_same_attrs,tuple_attrs=get_attr_to_same(join_conditions)
    # 接下来是处理每个连接属性的可能取值范围
    for jc in join_conditions:
        if isinstance(jc,str):
            continue
        else:  
            left,operator,right=jc
            all_join_attrs.add(left)
            all_join_attrs.add(right)
            add_attrs_in_dict(left.table,left.attr,table_join_attrs)
            add_attrs_in_dict(right.table,right.attr,table_join_attrs)
            # 上面是获取每个表对应的连接属性，下面是获取对应连接属性的可能取值范围
            domain1=get_join_domain(data_path,dataset,left,operator,right) 
            for same_attr in tuple_attrs:
                if left in same_attr:
                    res=base_attr_domain_sets.get(same_attr,0)
                else:
                    continue
                if res==0:
                    base_attr_domain_sets[same_attr]=domain1
                else:
                    new_res=res.intersection(set(domain1))
                    base_attr_domain_sets[same_attr]=new_res
    # 上面获取完毕每一个same_attr对应的可能取值，下面获取每个属性对应的same_attr字典
    attr_domain_set=dict()
    for k,v in base_attr_domain_sets.items():
        for ak in k:
            attr_domain_set[ak]=v
    # 下面获取经过分桶编码后的值域及其桶内占比
    for jc in join_conditions:
        if isinstance(jc,str):
            continue
        else:
            left,operator,right=jc
            base_domain=attr_domain_set[left]
            base_domain=list(base_domain)
            base_domain.sort()
            base_attr_domain_sets[attrs_to_same[left]]=base_domain
            value=get_original_bin_value(data_path,dataset,base_domain,left.table,left.attr)
            attrs_trans_domain[left]=value
            value=get_original_bin_value(data_path,dataset,base_domain,right.table,right.attr)
            attrs_trans_domain[right]=value
    return table_join_attrs,base_attr_domain_sets,attrs_to_same,attrs_trans_domain
def get_join_condition_domain(data_path,dataset,join_conditions):
    table_join_attrs=dict()
    attrs_domain_sets=dict()
    all_join_attrs=set()
    for jc in join_conditions:
        if isinstance(jc,str):
            continue
        else:
            # print(jc)
            left,operator,right=jc
            all_join_attrs.add(left)
            all_join_attrs.add(right)
            add_attrs_in_dict(left.table,left.attr,table_join_attrs)
            add_attrs_in_dict(right.table,right.attr,table_join_attrs)
            continue
            domain1=get_join_domain(data_path,dataset,left,operator,right)
            domain2=domain1.copy()
            left_res=attrs_domain_sets.get(left,0)
            if(left_res==0):
                left_res=set(domain1)
            else:
                left_res=left_res.intersection(set(domain1))
            # 将left_res的domain格式修改为类似select_domain_set的格式
            attrs_domain_sets[left]=left_res
            right_res=attrs_domain_sets.get(right,0)
            if(right_res==0):
                right_res=set(domain2)
            else:
                right_res=right_res.intersection(set(domain2))
            # 将right_res的domain格式修改为类似select_domain_set的格式
            attrs_domain_sets[right]=right_res
    for attr in all_join_attrs:
        # value=domain2binvalue(data_path,dataset,attrs_domain_sets[attr],attr)
        value=domain2binvalue(data_path,dataset,[],attr)
        attrs_domain_sets[attr]=value
    return table_join_attrs,attrs_domain_sets
def value_set2bin_prob(bin_value_dict,value):
    results=[]
    value=value.astype(int) # 此处是分桶之后的value
    value=pd.Series(value.flatten())
    value_count_dict=value.value_counts().to_dict()
    for value,counts in value_count_dict.items():
        total=sum(bin_value_dict[value].values()) # 当前桶中一共有几个数值
        results.append((value,counts/total))
    results.append(slice(0,0))
    return results
def get_join_domain(data_path,dataset,attr1:Attr,operator,attr2:Attr):
    if operator=='=':
        # 获取对应属性的可能取值范围
        res=load_attr_dict_or_decoder(data_path,dataset,attr1.table,attr1.attr)
        if res[0]:
            attr_dict=res[1]
            domain1=set(attr_dict.keys())
            res2=load_attr_dict_or_decoder(data_path,dataset,attr2.table,attr2.attr)
            attr_dict=res2[1]
            domain2=set(attr_dict.keys())
            join_domain=domain1.intersection(domain2)
        else: # 目前可确定id列基本都为decoder
            attr_decoder1,bin_value_dict1,unique_values1=res[1:]
            domain1=set(unique_values1)
            res2=load_attr_dict_or_decoder(data_path,dataset,attr2.table,attr2.attr)
            attr_decoder2,bin_value_dict2,unique_values2=res2[1:]
            domain2=set(unique_values2)
            join_domain=domain1.intersection(domain2)
            # original_value=np.array(list(join_domain)).reshape(-1,1)
        return join_domain
    else:
        raise NotImplementedError("Unsupported non-equal join condition!")
def load_attr_dict_or_decoder(data_path,dataset,table,attr):
    '''
    return True,attr_dict
    return False,attr_decoder,bin_value_dict,unique_values
    '''
    table_file=os.path.join(data_path,dataset,f'{table}.csv')
    dict_path=get_dict_path(table_file,attr)
    if os.path.exists(dict_path):
        with open(dict_path,'rb') as f:
            attr_dict=pickle.load(f)
        return True,attr_dict
    else:
        decoder_path=get_decoder_path(table_file,attr)
        bin_dict_path=get_bin_value_dict_path(table_file,attr)
        unique_path=get_unique_path(table_file,attr)
        with open(decoder_path,'rb') as f:
            attr_decoder=pickle.load(f)
        with open(bin_dict_path,"rb") as f:
            bin_value_dict=pickle.load(f)
        with open(unique_path,'rb') as f:
            unique_values=pickle.load(f)
        return False,attr_decoder,bin_value_dict,unique_values
def add_attrs_in_dict(table,attr,join_attr_dict):
    attr_set=join_attr_dict.get(table,set())
    attr_set.add(attr)
    join_attr_dict[table]=attr_set
def get_select_domain(data_path,dataset,select_conditions,tolower=False):
    table_select_attrs=dict()
    select_domain_sets=dict()
    for sc in select_conditions:
        if isinstance(sc,str):
            continue
        else:
            if tolower:
                sc[0].attr=sc[0].attr.lower()
            attr=sc[0].attr
            table=sc[0].table
            res=load_attr_dict_or_decoder(data_path,dataset,table,attr)
            if(res[0]):
                domain_set=handle_dict_condition(res[1],sc[1],sc[2])
            else: 
                domain_set=handle_decoder_condition(res[1],res[2],sc[1],sc[2])
                if(attr in table_select_attrs.get(table,set())):
                    select_domain_sets[sc[0]][-1]=slice_intersection(select_domain_sets[sc[0]][-1],domain_set[-1])
                    select_domain_sets[sc[0]].insert(1,domain_set[0])
                    continue
        add_attrs_in_dict(table,attr,table_select_attrs)
        select_domain_sets[sc[0]]=domain_set
    return table_select_attrs,select_domain_sets
# @profile
def sample_attr_value(bn,cols,domain_sets,attr,known_value,autosum=True):
    # 从MPN中获取对应的CPT
    domain_sets=domain_sets.copy()
    mpn=bn.list_MPN[cols.get_loc(attr.attr)]
    mpn.to(device)
    # 生成输入，输入为known_value
    #   初始化输入全为0
    x = torch.FloatTensor(1, bn.sum_cardinalities).zero_().to(device)
    s = torch.FloatTensor(x.shape).zero_().to(device)
    #   遍历known_value，将对应属性的对应下标替换为1
    index = 0
    this_node_do_not_has_value=True
    for node,value in known_value:
        if node==attr.attr:
            new_domain=None
            this_node_do_not_has_value=False
            original_domain=domain_sets[attr]
            # 将original_domain中属于对应当前属性value的值保留，其余的全部去除
            if in_slice(original_domain[-1],value):
                if isinstance(value,int):
                    new_domain=original_domain
                else:
                    new_domain=[(value.item(),1),slice(0,0)]
            else:
                for bin_index,prob in original_domain[:-1]:
                    if bin_index==value:
                        new_domain=[(bin_index,prob),slice(0,0)]
                        break
            if new_domain==None:
                new_domain=[(value,1),slice(0,0)]
            domain_sets[attr]=new_domain
        j=cols.get_loc(node)
        xx = torch.FloatTensor(1, bn.list_cardinalities[j]).to(device)
        xx.zero_()
        if isinstance(value,int):
            value=torch.LongTensor([value]).to(device).unsqueeze(1)
        # indices = torch.LongTensor(value).unsqueeze(1)
        # xx.scatter_(1, indices, 1)
        xx.scatter_(1, value, 1)
        # print(xx)
        index += 1
        x[:, sum(bn.list_cardinalities[:j]):sum(bn.list_cardinalities[:j+1])] = xx
    # 获取输出
    output_m,output_s=mpn((x,s))
    # 还差一步，根据查询条件，对概率分布进行去除重归一化
    if output_m.isnan().any(): # 治标不治本！！！！！！！！
        print('nan')
        output_m=torch.where(torch.isnan(output_m), torch.tensor(1/1001), output_m)
    prob_distribution=torch.zeros_like(output_m)
    if isinstance(domain_sets[attr][-1],tuple):
        for t in domain_sets[attr][-1]:
            prob_distribution[0,t]=output_m[0,t]
    else:
        prob_distribution[0,domain_sets[attr][-1]]=output_m[0,domain_sets[attr][-1]]
    # print(prob_distribution.shape,prob_distribution)
    for bin_index,prob in domain_sets[attr][:-1]:
        prob_distribution[0,bin_index]=output_m[0,bin_index]*prob
    withoutnan_prob=torch.zeros((prob_distribution.shape[0],bin_nums)).to(device)
    withoutnan_prob[0,:bin_nums]=prob_distribution[0,:bin_nums]
    # renormal_prob=prob_distribution/prob_distribution.sum()
    renormal_prob=withoutnan_prob/withoutnan_prob.sum()
    # print(prob_distribution.shape,prob_distribution)
    # 根据输出概率采样得到新的值
    if this_node_do_not_has_value:
        sample_result = torch.multinomial(renormal_prob, num_samples=1, replacement=True)
    else:
        sample_result=None
    # print(sample_result)
    # 返回采样结果和输出的求和
    del x, s, output_m, output_s, renormal_prob  # 删除不再需要的对象
    # torch.cuda.empty_cache()  # 清理CUDA缓存
    # # gc.collect()
    # collected = gc.collect()
    # if collected!=0:
    #     print(f"垃圾回收器回收了{collected}个对象! ")
    if autosum:
        return sample_result,prob_distribution.sum()
    else:
        return sample_result,prob_distribution[0,:bin_nums]
def handle_dict_condition(attr_dict, operator, value):
    all_keys = attr_dict.keys()
    filtered_values = []
    if operator == '=':
        filtered_values = [attr_dict[value]] if value in attr_dict else []
    elif operator == '>':
        filtered_values = [attr_dict[k] for k in all_keys if k > value]
    elif operator == '<':
        filtered_values = [attr_dict[k] for k in all_keys if k < value]
    elif operator == '>=':
        filtered_values = [attr_dict[k] for k in all_keys if k >= value]
    elif operator == '<=':
        filtered_values = [attr_dict[k] for k in all_keys if k <= value]
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return filtered_values
def in_slice(slice,value):
    if slice.start==None:
        start=0
    else:
        start=slice.start
    if slice.stop==None:
        stop=float('inf')
    else:
        stop=slice.stop
    return start <= value <= stop
def slices_intersection(slices1,slices2):
    a=isinstance(slices1,slice)
    b=isinstance(slices2,slice)
    if a and b:
        return slice_intersection(slices1,slices2)
    if not a and b:
        new_s1=[slice_intersection(s,slices2) for s in slices1]
        return tuple(new_s1)
    if a and not b:
        new_s2=[slice_intersection(slices1,s) for s in slices2]
    # 两个均为tuple
    final_s=[]
    for s1 in slices1:
        for s2 in slices2:
            temp_s=slice_intersection(s1,s2)

def slice_merge(slice1,slice2):
    # 获取两个切片的起始和结束索引
    start1, stop1 = slice1.start or 0, slice1.stop
    start2, stop2 = slice2.start or 0, slice2.stop

def slice_intersection(slice1, slice2):
    # 获取两个切片的起始和结束索引
    start1, stop1 = slice1.start or 0, slice1.stop
    start2, stop2 = slice2.start or 0, slice2.stop
    # 计算交集的起始和结束索引
    start_max = max(start1, start2)
    stop_min = min(stop1, stop2)
    # 如果交集有效（即起始索引小于结束索引），则返回交集的 slice 对象
    if start_max < stop_min:
        return slice(start_max, stop_min)
    else:
        # 如果没有交集，返回一个表示空切片的slice对象
        return slice(0,0)
def slice_remove_bin(s,index):
    # 从slice切片中去除index取值
    start=s.start or 0
    stop=s.stop
    if start>index:
        return s
    if stop<=index:
        return s
    return (slice(start,index),slice(index+1,stop))

def handle_decoder_condition(attr_decoder, bin_value_dict, operator, value):
    # 使用 KBinsDiscretizer 转换输入值
    is_date_type=False
    if("::timestamp" in value):
        is_date_type=True
        value=value.split('::timestamp')[0].replace('\'','')
        value=pd.to_datetime(value).timestamp()
        value=int(value)
    else:
        value=float(value)
    transformed_value = attr_decoder.transform(np.array([value]).reshape(-1, 1))
    bin_index = transformed_value[0, 0] # value对应桶
    bin_index = int(bin_index)
    bin_edges = attr_decoder.bin_edges_[0]
    # 桶区间大小，长度为总桶数+1
    results = [] # 第一个为(bin_index,p)p为占的比例，后续的桶占比例均为1
    bin_value_nums=sum(bin_value_dict[bin_index].values())
    if(is_date_type):
        bin_prob_values=[int(k) for k in bin_value_dict[bin_index].keys()]
    else:
        bin_prob_values=[float(k) for k in bin_value_dict[bin_index].keys()]
    nan_bin_index=int(attr_decoder.transform(np.array([-1]).reshape(-1,1))[0][0])
    if operator == '=':
        results.append((bin_index,bin_value_dict[bin_index][value]/bin_value_nums))
        results.append(slice(0,0))
    elif operator == '<':
        nums=0
        for v in bin_prob_values:
            if v<value:
                nums+=bin_value_dict[bin_index][v]
        results.append((bin_index,nums/bin_value_nums))
        # results.append((bin_index,(value-bin_edges[bin_index])/(bin_edges[bin_index+1]-bin_edges[bin_index])))
        # results.append(slice_remove_bin(slice(bin_index)),nan_bin_index)
        results.append(slice(bin_index))
        # for i in range(bin_index):
            # results.append(i)
    elif operator == '>':
        nums=0
        for v in bin_prob_values:
            if v>value:
                nums+=bin_value_dict[bin_index][v]
        results.append((bin_index,nums/bin_value_nums))
        # results.append((bin_index,(bin_edges[bin_index+1]-value)/(bin_edges[bin_index+1]-bin_edges[bin_index])))
        # results.append(slice_remove_bin(slice(bin_index+1,len(bin_edges)-1),nan_bin_index))
        results.append(slice(bin_index+1,len(bin_edges)-1))
        # for i in range(bin_index + 1, len(bin_edges) - 1):
            # results.append(i)
    elif operator == '<=':
        nums=0
        for v in bin_prob_values:
            if v<=value:
                nums+=bin_value_dict[bin_index][v]
        results.append((bin_index,nums/bin_value_nums))
        # results.append(slice_remove_bin(slice(bin_index),nan_bin_index))
        results.append(slice(bin_index))
        # for i in range(bin_index + 1):
            # results.append(i)
    elif operator == '>=':
        nums=0
        for v in bin_prob_values:
            if v>=value:
                nums+=bin_value_dict[bin_index][v]
        results.append((bin_index,nums/bin_value_nums))
        # results.append((bin_index,((bin_edges[bin_index+1]-value))/(bin_edges[bin_index+1]-bin_edges[bin_index])))
        # results.append(slice_remove_bin(slice(bin_index+1,len(bin_edges)-1),nan_bin_index)
        results.append(slice(bin_index+1,len(bin_edges)-1))
        # for i in range(bin_index, len(bin_edges) - 1):
            # results.append(i)
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    return results