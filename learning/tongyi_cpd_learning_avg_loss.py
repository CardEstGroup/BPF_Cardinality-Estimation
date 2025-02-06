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
    all_train_loss=0
    node_nums=BN.num_nodes
    for batch_idx, data in enumerate(list_tarin_dl_tensor):
        train_loss_all = 0
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
        best_iter = -1
        best_loss = np.full((node_nums),np.inf)
        train_loss = np.inf
        convergence_iter_num = 0
        epoch_loss=np.zeros((node_nums))
        for epoch in range(1, para['parameter_learning_max_iter_num'] + 1):
            optimizer.zero_grad()
            avg_loss = 0
            for i in range(node_nums):
                model = BN.list_MPN[i]
                model.train()
                loss = model.loss(data_m, data_s, targets[i],w,prior)
                loss.backward()
                epoch_loss[i]=loss.item()
                # avg_loss+=loss.item()/node_nums
                # avg_loss+=loss
            # total_loss.backward()
            optimizer.step()
            train_loss_all += avg_loss
            train_loss=avg_loss
            if epoch == 1:
                continue
            if (epoch_loss < best_loss).sum()!=0:  # 训练损失小于当前最优损失
                for k in range(node_nums):
                    best_loss[k]=epoch_loss[k]
                best_iter = epoch
                convergence_iter_num = 0
            else:  # 训练损失不小于当前最优损失
                convergence_iter_num += 1
            if epoch == para['parameter_learning_max_iter_num'] or convergence_iter_num == para['parameter_learning_convergence_max_iter_num']:
                print(f'Train Epoch: {epoch}, best_loss: {best_loss}, best_iter: {best_iter}, convergence_iter_num: {convergence_iter_num}')
                break
        avg_train_loss=train_loss_all/epoch
        all_train_loss+=avg_train_loss
    likelihood = -1 * all_train_loss
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