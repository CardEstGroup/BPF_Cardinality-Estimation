# imports
import os
import vaex
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from my_model.BN import BN
from basic_info import BasicInfo
from query_parse import parse_sql
from data.loaddata import my_load_data,my_load_tarin_dl
from preprocess_csv import csv_process,csv_bin_process,csv_exclude_nan
from learning.tongyi_cpd_learning import CPD_train_multi_batch
from get_path import get_bn_save_path,get_basic_info_save_path

# hyper parameters
device='cuda'
mi_threshold=0.7
fre_threshold=0.7
# 0.2,0.2,0.2,0.4 均分后为0.25，高于0.25的0.8倍0.2的节点为候选子节点
bin_nums=1900
base_para = {
    'device': "cuda",
    #基于NPM的MPN比基于普通NN（即vanillaNN）收敛更稳定，但它们的输入输出无本质区别
    'type_MPN':"NPN",
    # 'type_MPN':"vanillaNN",
    'num_nodes': 1, # BNML变量个数
    'list_cardinalities': [2], # 列表: 1表示连续变量的势，1~n表示离散变量的势
    'train_ratio': 1.0, # 训练集的比例
    'batch_size': 9999999, # 9999999 9个屁，40325就已经放不下了
    'lr': 1e-2, # 学习率
    'activation' : 'sigmoid', # MPN的激活函数
    'num_neuron_latent_layer': [64, 64], # MPN隐藏层的神经元个数
    'parameter_learning_max_iter_num': 2000,  # 参数学习最大迭代次数
    'parameter_learning_convergence_max_iter_num': 100, # 用于早停的收敛次数
    'structure_learning_max_iter_num': 10, # 结构学习最大迭代次数
    'print_interval': 100, # MPN训练过程中，相关信息的打印频次，如100次训练打印1次
}
exact_para = {
    'device': "cuda",
    #基于NPM的MPN比基于普通NN（即vanillaNN）收敛更稳定，但它们的输入输出无本质区别
    'type_MPN':"NPN",
    # 'type_MPN':"vanillaNN",
    'num_nodes': 1, # BNML变量个数
    'list_cardinalities': [2], # 列表: 1表示连续变量的势，1~n表示离散变量的势
    'train_ratio': 1.0, # 训练集的比例
    'batch_size': 40000, # 9999999 9个屁，40325就已经放不下了
    'lr': 1e-1, # 学习率
    'activation' : 'sigmoid', # MPN的激活函数
    'num_neuron_latent_layer': [64, 64], # MPN隐藏层的神经元个数
    'parameter_learning_max_iter_num': 500,  # 参数学习最大迭代次数
    'parameter_learning_convergence_max_iter_num': 200, # 用于早停的收敛次数
    'structure_learning_max_iter_num': 10, # 结构学习最大迭代次数
    'print_interval': 200, # MPN训练过程中，相关信息的打印频次，如100次训练打印1次
}
quick_para = {
    'device': "cuda",
    #基于NPM的MPN比基于普通NN（即vanillaNN）收敛更稳定，但它们的输入输出无本质区别
    'type_MPN':"NPN",
    # 'type_MPN':"vanillaNN",
    'num_nodes': 1, # BNML变量个数
    'list_cardinalities': [2], # 列表: 1表示连续变量的势，1~n表示离散变量的势
    'train_ratio': 1.0, # 训练集的比例
    'batch_size': 40000, # 9999999 9个屁，40325就已经放不下了
    'lr': 1e-1, # 学习率
    'activation' : 'sigmoid', # MPN的激活函数
    'num_neuron_latent_layer': [64, 64], # MPN隐藏层的神经元个数
    'parameter_learning_max_iter_num': 20,  # 参数学习最大迭代次数
    'parameter_learning_convergence_max_iter_num': 200, # 用于早停的收敛次数
    'structure_learning_max_iter_num': 10, # 结构学习最大迭代次数
    'print_interval': 100, # MPN训练过程中，相关信息的打印频次，如100次训练打印1次
}
para = {
    'device': "cuda",
    #基于NPM的MPN比基于普通NN（即vanillaNN）收敛更稳定，但它们的输入输出无本质区别
    'type_MPN':"NPN",
    # 'type_MPN':"vanillaNN",
    'num_nodes': 1, # BNML变量个数
    'list_cardinalities': [2], # 列表: 1表示连续变量的势，1~n表示离散变量的势
    'train_ratio': 1.0, # 训练集的比例
    'batch_size': 15000, # 9999999 9个屁，40325就已经放不下了
    'lr': 1e-2, # 学习率
    'activation' : 'sigmoid', # MPN的激活函数
    'num_neuron_latent_layer': [64, 64], # MPN隐藏层的神经元个数
    'parameter_learning_max_iter_num': 5,  # 参数学习最大迭代次数
    'parameter_learning_convergence_max_iter_num': 200, # 用于早停的收敛次数
    'structure_learning_max_iter_num': 10, # 结构学习最大迭代次数
    'print_interval': 200, # MPN训练过程中，相关信息的打印频次，如100次训练打印1次
}
# 主流程
def train_model(query_file,table_file):
    """
    整个从csv学习MPN的主要流程
    """
    # 初始化
    table_name=os.path.basename(table_file).split('.')[0]
    headdf=pd.read_csv(table_file,escapechar='\\',chunksize=10)
    headdf=headdf.__next__()
    cols=headdf.columns
    # 统计查询负载中节点出现频率
    sort_fre=count_attr_fre(query_file)
    # 确定拓扑序
    attr_order=[]
    for sf in sort_fre:
        table,attr=sf[0].split('.')
        if(table==table_name):
            attr_order.append(attr)
    for col in cols:
        if col not in attr_order:
            attr_order.append(col)
    # 统计查询负载中任两节点共同出现频率
    qpm=count_attrs_fre(query_file,table_name,attr_order)
    # 通过vaex估算任意两列间互信息，对角线为信息熵
    # 得到MIM（Mutual Information Matrix）
    new_table_file=csv_process(table_file)
    mim=get_mim(new_table_file)
    # 预处理数据表
    # new_table_file,dict_dir=csvpreprocess(table_file)
    new_table_file,pkl_dir=csv_exclude_nan(table_file,k=bin_nums)
    # 预处理数据
    data=pd.read_csv(new_table_file,escapechar='\\')
    data_lens=data.shape[0]
    num_nodes=len(attr_order)
    list_cardinalities=[]
    for attr in cols:
        cardinality=data[attr].max()+1
        list_cardinalities.append(cardinality)
    # 初始化DAG结构
    init_edges = np.zeros((num_nodes, num_nodes), dtype=int, order='C')  # 初始边的邻接矩阵
    old_edges = np.zeros((num_nodes, num_nodes), dtype=int, order='C')  # 可能边的约束矩阵
    old_edges2 = np.zeros((num_nodes, num_nodes), dtype=int, order='C')  # 可能边的约束矩阵
    constraint_may_edges = np.zeros((num_nodes, num_nodes), dtype=int, order='C')  # 可能边的约束矩阵
    # 通过拓扑序及QPM矩阵限制可能加边
    # 构建拓扑序和原始属性序的映射    
    index_mapping = {attr_order.index(attr): index for index,attr in enumerate(cols)}
    for i in range(num_nodes-1): # 邻接矩阵按拓扑序排列
        # 从可能的子节点中选择前int(0.5k)个作为候选子节点
        thre=1/(num_nodes-i-1)*fre_threshold
        for j in range(i+1,num_nodes):
            # print(qpm[i][j],qpm[i][i],qpm[i][j]/qpm[i][i],thre)
            newi=index_mapping[i]
            newj=index_mapping[j]
            if(qpm[i][i]!=0):
                fre_partition=qpm[i][j]/qpm[i][i]
                # if(fre_partition>thre):
                if(fre_partition>fre_threshold):
                    old_edges[i][j]=1
                    # constraint_may_edges[newi][newj]=1
                # print(mim[i][j]/mim[j][j],mi_threshold)
            if(fre_partition>thre/2+0.5): # 查询频率过高
                constraint_may_edges[newi][newj]=1
                continue
            if(mim[j][j]==0 or mim[i][j]==0): # 跳过未估算出互信息的值
                continue
            mi_partition=mim[i][j]/mim[j][j]
            if(mi_partition>mi_threshold):
                old_edges2[i][j]=1
            if(mi_partition>mi_threshold/2+0.5): # 互信息过高
                constraint_may_edges[newi][newj]=1
                continue
            if(old_edges2[i][j] and old_edges[i][j]):
                # 将邻接矩阵更换为按属性序的邻接矩阵
                constraint_may_edges[newi][newj]=1
    # 基础参数学习
    # bn = BN(num_nodes, list_cardinalities, init_edges, constraint_may_edges, para)
    bn = BN(num_nodes, list_cardinalities, constraint_may_edges, init_edges, para)
    # 直接通过互信息和查询偏好确定网络结构
    bn.create_CPD(device)
    # check mpn is in cuda
    # print('MPN is in cuda: ')
    # for i in range(bn.num_nodes):  # 0 ~ num_nodes-1
            # print(next(bn.list_MPN[i].parameters()).is_cuda)  # True
            # print(bn.list_MPN[i])  # True
    data_file_path = new_table_file
    train_dl, test_dl, trainLen, testLen = my_load_data(para, test_batch_size=None, shuffle=True, data_file_path=data_file_path)
    # print('trainLen:' + str(trainLen))
    # print('testLen:' + str(testLen))
    para['train_size'] = trainLen
    para['test_size'] = testLen
    # 将加载的数据[1, 1, 2]转化为onehot形式
    # [tensor([[1., 0.],
    #          [1., 0.],
    #          [0., 1.]])]
    # 从而通过训练该变量对应的MPN，MPN输出的CPT为[0.67, 0.33]
    # list_tarin_dl_tensor = []
    # for batch_idx, data in enumerate(train_dl):  # batch_idx: 0~n
        # input = my_load_tarin_dl(data, para, bn)
        # list_tarin_dl_tensor.append(input)
    para['print_interval'] = 10000 # 修改：结构学习在调用参数学习时，参数学习的打印间隔
    # 结构学习初始化：在结构学习前，先在初始结构上学习每个节点的CPT，进而得到每个节点的FBIC评分
    # for i in range(0, bn.num_nodes):
        # bn.list_BIC_loglikelihood[i] = CPD_train_multi_batch(bn, i, para, train_dl)  # 目前学习过程不涉及测试数据，即test_dl=None，可忽略test_dl
        # bn.list_BIC_penalty[i] = bn.independent_parameters_node_i(i) / 2 * np.log(para['train_size'])
        # bn.list_BIC[i] = bn.list_BIC_loglikelihood[i] - bn.list_BIC_penalty[i]
    sumloglikelihood=CPD_train_multi_batch(bn, para, train_dl)
    bn_file_path=get_bn_save_path(table_file)
    torch.save(bn,bn_file_path)
    with open(get_basic_info_save_path(table_file),'wb') as f:
        basic_info=BasicInfo(data_lens,attr_order,cols)
        pickle.dump(basic_info,f)
    return True
# 统计查询负载中节点出现频率
def count_attr_fre(query_file):
    fre_dict={}
    with open(query_file,'r') as f:
        for line in f:
            table_alias,select_conditions,join_conditions=parse_sql(line)
            # table_alias 不对
            select_name=set()
            for sc in select_conditions:
                if(isinstance(sc,str)):
                    continue
                attr_name=table_alias[sc[0].table]+'.'+sc[0].attr
                select_name.add(attr_name)
            for sname in select_name:
                fre_dict[sname]=fre_dict.get(sname,0)+1
            join_name=set()
            for jc in join_conditions:
                if(isinstance(jc,str)):
                    continue
                attr_name=table_alias[jc[0].table]+'.'+jc[0].attr
                attr_name2=table_alias[jc[2].table]+'.'+jc[2].attr
                join_name.add(attr_name)
                join_name.add(attr_name2)
            for jname in join_name:
                fre_dict[jname]=fre_dict.get(jname,0)+1
    return sorted(fre_dict.items(), key = lambda kv:(kv[1]),reverse=True) 

# 统计查询负载中任两节点共同出现频率
def count_attrs_fre(query_file,table_name,attr_order):
    node_lens=len(attr_order)
    qpm=np.zeros((node_lens,node_lens)) # Query Preference Martix
    outf=open('res.txt','w')
    with open(query_file,'r') as f:
        for line in f:
            line_attrs=set()
            table_alias,select_conditions,join_conditions=parse_sql(line)
            # table_alias 不对 
            for sc in select_conditions:
                if(isinstance(sc,str)):
                    continue
                if(table_alias[sc[0].table]==table_name):
                    line_attrs.add(sc[0].attr)
            for jc in join_conditions:
                if(isinstance(jc,str)):
                    continue
                if(table_alias[jc[0].table]==table_name):
                    line_attrs.add(jc[0].attr)
                if(table_alias[jc[2].table]==table_name):
                    line_attrs.add(jc[2].attr)
            line_attrs=list(line_attrs)
            if(len(line_attrs)!=0):
                outf.write(str(line_attrs)+'\n')
            if(len(line_attrs)==1):
                index=attr_order.index(line_attrs[0])
                qpm[index][index]+=1
            else:
                for i in range(len(line_attrs)):
                    for j in range(len(line_attrs)):
                        x=attr_order.index(line_attrs[i])
                        y=attr_order.index(line_attrs[j])
                        qpm[x][y]+=1
    outf.close()
    return qpm
# 得到MIM（Mutual Information Matrix）
def get_mim(new_table_file):
    table=vaex.read_csv(new_table_file)
    col_names=table.get_column_names()
    col_len=len(col_names)
    mim=np.zeros((col_len,col_len))
    for i in range(col_len):
        for j in range(i,col_len):
            mim[i][j]=table.mutual_information(col_names[i],col_names[j])
            mim[j][i]=mim[i][j]
    return mim

if __name__=='__main__':
    # const
    job='sqls/job_sub_query.sql'
    ceb='sqls/stats_sub_query.sql'
    table_file='datasets/imdb/movie_info_idx.csv'
    table_file='test_data/users.csv'
    table_file='datasets/stats/posts.csv'
    # basic init
    query_file=job
    query_file=ceb

    # main(query_file,table_file)
    res=train_model(query_file,table_file)
    if res:
        print("训练并保存成功！")
