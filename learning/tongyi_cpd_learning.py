import torch
import numpy as np
import torch.optim as optim
from data.loaddata import my_load_tarin_dl

def CPD_train_multi_batch(BN, para, list_tarin_dl_tensor, test_dl=None):
    # 将所有MPN的参数加到优化器中
    params_list = [p for model in BN.list_MPN for p in model.parameters()]
    optimizer = optim.Adam(params_list, lr=para['lr'], weight_decay=0)
    w = None
    prior = None
    node_nums=BN.num_nodes
    best_iter = np.full((node_nums),-1)
    best_loss = np.full((node_nums),np.inf)
    convergence_iter_num = np.zeros((node_nums))
    breaki = np.zeros((node_nums))
    for epoch in range(1, para['parameter_learning_max_iter_num'] + 1):
        epoch_loss=np.zeros((node_nums))
        for batch_idx, data in enumerate(list_tarin_dl_tensor):
            input = my_load_tarin_dl(data, para, BN)
            optimizer.zero_grad()
            for i in range(BN.num_nodes):
                data_m = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(
                    para['device'])  # 每行数据为：当前节点的父节点取值的onehot
                data_s = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(para['device'])  # 总是为0，可忽略
                target = torch.FloatTensor(len(input), BN.list_cardinalities[i]).zero_().to(
                    para['device'])  # 每行数据为：当前节点的取值的onehot
                for j in range(BN.num_nodes):
                    if BN.edges[j][i] == 1:
                        data_m[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])] = input[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])]
                    if i == j:
                        target = input[:, sum(BN.list_cardinalities[:i]):sum(BN.list_cardinalities[:i + 1])]
                if(breaki[i]==1):
                    continue
                model = BN.list_MPN[i]
                model.train()
                loss = model.loss(data_m, data_s, target,w,prior)
                loss.backward()
                epoch_loss[i]=loss.item()
            optimizer.step()
        if epoch == 1:
            continue
        for k in range(node_nums):
            if breaki[k]==1:
                continue
            if epoch_loss[k] < best_loss[k]: # 训练损失小于当前最优损失
                best_loss[k]=epoch_loss[k]
                best_iter[k]=epoch
                convergence_iter_num[k]=0
            else:
                convergence_iter_num[k]+=1
            if convergence_iter_num[k] == para['parameter_learning_convergence_max_iter_num'] or epoch == para['parameter_learning_max_iter_num']:
                print(f'Node {k+1}, Train Epoch: {epoch}, best_loss: {best_loss[k]:.2f}, best_iter: {best_iter[k]}, convergence_iter_num: {convergence_iter_num[k]}')
                breaki[k]=1
        if(breaki.sum()==node_nums):
            break
    likelihood = -1 * epoch_loss
    return likelihood
def train_node(i, data_m, data_s, targets, optimizer, epoch, para, BN):
    w = None
    prior = None
    model = BN.list_MPN[i]
    model.train()
    loss = model.loss(data_m, data_s, targets[i],w,prior)
    # if epoch % para['print_interval'] == 0:
    #     print(f'Node: {i+1}, Train Epoch: {epoch}, Batch loss: {loss.item():.4f}')
    return loss

def CPD_learning(epoch, BN, optimizer, para, list_tarin_dl_tensor, test_dl=None):
    train_loss_all = 0
    for batch_idx, data in enumerate(list_tarin_dl_tensor):
        input = my_load_tarin_dl(data, para, BN)
        data_m = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(para['device'])
        data_s = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(para['device'])
        targets = [torch.FloatTensor(len(input), BN.list_cardinalities[i]).zero_().to(para['device']) for i in range(BN.num_nodes)]

        for j in range(BN.num_nodes):
            for i in range(BN.num_nodes):
                if BN.edges[j][i] == 1:
                    data_m[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])] = input[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])]
                if i == j:
                    targets[i] = input[:, sum(BN.list_cardinalities[:i]):sum(BN.list_cardinalities[:i + 1])]
        optimizer.zero_grad()
        total_loss = 0
        for i in range(BN.num_nodes):
            w = None
            prior = None
            model = BN.list_MPN[i]
            model.train()
            loss = model.loss(data_m, data_s, targets[i],w,prior)
            total_loss+=loss
        total_loss.backward()
        optimizer.step()
        train_loss_all += total_loss
    return train_loss_all