import torch
import numpy as np
import torch.optim as optim
from data.loaddata import my_load_tarin_dl
# np.set_printoptions(threshold=np.inf) # 当数组元素过多时，避免输出数组时出现省略号

def CPD_train_multi_batch(BN, i, para, list_tarin_dl_tensor, test_dl = None):
    # 将MPN的参数加到优化器中
    # new_para=para.copy()
    params_list = []
    if i!= None:
        params_list.append(BN.list_MPN[i].parameters())
    optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr'], weight_decay=0) # lr
    # print(optimizer)
    # exit(0)

    best_iter = -1
    best_loss = np.inf
    train_loss = np.inf
    convergence_iter_num = 0
    # max_epoch=para['parameter_learning_max_iter_num'] + 1
    for epoch in range(1, para['parameter_learning_max_iter_num'] + 1):
        train_loss = CPD_learning(epoch, BN, i, optimizer, para, list_tarin_dl_tensor, test_dl)
        # if epoch % para['print_interval']==0:
            # print(f'Node: {i+1} Epoch: {epoch}/{max_epoch} Loss: {train_loss}')
        # early stopping 早停
        if epoch == 1:
            continue
        if train_loss < best_loss: # 训练损失小于当前最优损失
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else: # 训练损失不小于当前最优损失
            convergence_iter_num += 1
            # lr = new_para['lr'] * 0.1 # 学习率自适应下降
            # new_para['lr'] = lr
            # print(f"Down learning rate to {lr}")
            # if(new_para['lr'] > 1e-2):
            #     convergence_iter_num = 0
        # if (convergence_iter_num>para['parameter_learning_convergence_max_iter_num']-1):
        #     lr = new_para['lr'] * 0.1 # 学习率自适应下降
        #     new_para['lr'] = lr
        #     print(f"Down learning rate to {lr}")
        #     if(new_para['lr'] > 1e-3):
        #         convergence_iter_num = 0
        if epoch == para['parameter_learning_max_iter_num'] or convergence_iter_num == para['parameter_learning_convergence_max_iter_num']:
            print(' Node: {}, Train Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(i+1, epoch, best_loss, best_iter, convergence_iter_num))
            break
    likelihood = -1 * train_loss
    return likelihood


def CPD_learning(epoch, BN, i, optimizer, para, list_tarin_dl_tensor, test_dl = None):
    train_loss_all = 0

    # for batch_idx in range(len(list_tarin_dl_tensor)):  # batch_idx: 0~n
    #     # 构建MPN的训练数据
    #     input = list_tarin_dl_tensor[batch_idx]
    max_batch_times=para.get('max_batch_times',0)
    times=0
    for batch_idx, data in enumerate(list_tarin_dl_tensor):  # batch_idx: 0~n
        times+=1
        if max_batch_times!=0:
            if(times>max_batch_times):
                break
        input = my_load_tarin_dl(data, para, BN)
        # print(input)
        # print(input.shape)
        # exit(0)

        data_m = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(
            para['device'])  # 每行数据为：当前节点的父节点取值的onehot
        data_s = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(para['device'])  # 总是为0，可忽略
        target = torch.FloatTensor(len(input), BN.list_cardinalities[i]).zero_().to(
            para['device'])  # 每行数据为：当前节点的取值的onehot

        for j in range(BN.num_nodes):  # 0 ~ num_nodes-1
            if BN.edges[j][i] == 1:  # 给父节点对应的行赋值
                data_m[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])] \
                    = input[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])]
            if i == j:  # 给当前节点输出赋值
                target = input[:, sum(BN.list_cardinalities[:i]):sum(BN.list_cardinalities[:i + 1])]
        # print(data_m)
        # print(data_s)
        # print(target)
        # exit(0)

        # 使用数据训练MPN，然后将每批数据的loss相加，得到当前节点的似然函数值
        train_loss = MPN_learning(BN, i, BN.list_MPN[i], optimizer, epoch, para, data_m, data_s, target)
        train_loss_all += train_loss
        # print(data_m)
        # print(data_s)
        # print(target)
        # exit(0)
    # exit(0)
    return train_loss_all


def CPD_train_no_batch(BN, i, para, list_tarin_dl_tensor, test_dl = None):
    # 将MPN的参数加到优化器中
    params_list = []
    if i!= None:
        params_list.append(BN.list_MPN[i].parameters())
    optimizer = optim.Adam([{'params': p} for p in params_list], lr = para['lr'], weight_decay=0) # lr
    # print(optimizer)
    # exit(0)

    # 构建MPN的训练数据
    input = list_tarin_dl_tensor[0]
    # print(input)
    # print(input.shape)
    # exit(0)
    data_m = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(
        para['device'])  # 每行数据为：当前节点的父节点取值的onehot
    data_s = torch.FloatTensor(len(input), BN.sum_cardinalities).zero_().to(
        para['device'])  # 总是为0，可忽略
    target = torch.FloatTensor(len(input), BN.list_cardinalities[i]).zero_().to(
        para['device'])  # 每行数据为：当前节点的取值的onehot

    for j in range(BN.num_nodes):  # 0 ~ num_nodes-1
        if BN.edges[j][i] == 1:  # 给父节点对应的行赋值
            data_m[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])] \
                = input[:, sum(BN.list_cardinalities[:j]):sum(BN.list_cardinalities[:j + 1])]
        if i == j:  # 给当前节点输出赋值
            target = input[:, sum(BN.list_cardinalities[:i]):sum(BN.list_cardinalities[:i + 1])]
    # print(data_m)
    # print(data_s)
    # print(target)
    # exit(0)

    best_iter = -1
    best_loss = np.inf
    train_loss = np.inf
    convergence_iter_num = 0
    for epoch in range(1, para['parameter_learning_max_iter_num'] + 1):
        # 使用数据训练MPN，然后将每批数据的loss相加，得到当前节点的似然函数值
        train_loss =  MPN_learning(BN, i, BN.list_MPN[i], optimizer, epoch, para, data_m, data_s, target)
        # early stopping 早停
        if epoch == 1: 
            continue
        if train_loss < best_loss: # 训练损失小于当前最优损失
            best_loss = train_loss
            best_iter = epoch
            convergence_iter_num = 0
        else: # 训练损失不小于当前最优损失
            convergence_iter_num += 1
        if epoch == para['parameter_learning_max_iter_num'] or convergence_iter_num == para['parameter_learning_convergence_max_iter_num']:
            print('Node: {}, Train Epoch: {}, best_loss: {:.2f} best_iter: {} convergence_iter_num: {}'.format(i+1, epoch, best_loss, best_iter, convergence_iter_num))
            break
    likelihood = -1 * train_loss
    return likelihood

def MPN_learning(BN, i, model, optimizer, epoch, para, data_m, data_s, target):
    # 如果模型中有Batch Normalization层和Dropout，需要在训练时添加model.train()。
    # 保证Batch Normalization层能够用到每一批数据的均值和方差，保证Dropout是随机取一部分网络连接来训练更新参数
    model.train()
    train_loss = 0
    print_output = 0
    w = None
    prior = None

    optimizer.zero_grad()  # 清空过往梯度
    loss = model.loss(data_m, data_s, target, w, prior, print_output)
    loss.backward() # 反向传播，计算当前梯度
    optimizer.step() # 根据梯度更新网络参数
    train_loss += loss.item()

    # if epoch % para['print_interval'] == 0:
    #     print('Node: {}, Train Epoch: {}, Train loss: {:.4f}, Batch size: {}'.format(i+1, epoch, train_loss , len(data_m)))
    #     output = model((data_m, data_s))
    #     print(output[0]) # 输出的均值作为CPT的取值
    #     # print(output[1]) # 输出的方法可忽视
    #     # exit(0)

    return train_loss



