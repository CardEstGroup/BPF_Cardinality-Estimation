import pandas as pd
from torch.utils.data import random_split
import torch

def my_load_data(para, test_batch_size, shuffle = True, data_file_path = None):
    train_loader = pd.read_table(data_file_path, sep=',', engine='python').values
    trainLen = round(para['train_ratio'] * len(train_loader))
    testLen = len(train_loader) - trainLen

    if testLen != 0:
        train, test = random_split(train_loader, [trainLen, len(train_loader) - trainLen])
    else:
        train = train_loader
        test = None

    train_dl = torch.utils.data.DataLoader(train, batch_size=para['batch_size'], shuffle=shuffle, pin_memory=True)
    test_dl = None
    if testLen != 0:
        test_dl = torch.utils.data.DataLoader(test, batch_size=testLen, shuffle=shuffle, pin_memory=True)

    return train_dl, test_dl, trainLen, testLen

def load_data(para, test_batch_size, shuffle = True, data_file_path = None):
    train_loader = pd.read_table(data_file_path, sep=',', engine='python', header=None).values
    trainLen = round(para['train_ratio'] * len(train_loader))
    testLen = len(train_loader) - trainLen

    if testLen != 0:
        train, test = random_split(train_loader, [trainLen, len(train_loader) - trainLen])
    else:
        train = train_loader
        test = None

    train_dl = torch.utils.data.DataLoader(train, batch_size=para['batch_size'], shuffle=shuffle, pin_memory=True)
    test_dl = None
    if testLen != 0:
        test_dl = torch.utils.data.DataLoader(test, batch_size=testLen, shuffle=shuffle, pin_memory=True)

    return train_dl, test_dl, trainLen, testLen


def my_load_tarin_dl(data, para, bn):
    x = torch.FloatTensor(len(data[:, 0]), bn.sum_cardinalities).zero_()
    index = 0
    for j in range(bn.num_nodes): # 0 ~ num_nodes-1
        # xx = torch.FloatTensor(len(data[:,0]), bn.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1), 1)
        xx = torch.FloatTensor(len(data[:,0]), bn.list_cardinalities[j])
        xx.zero_()
        indices = data[:, index].to(int).unsqueeze(1)
        xx.scatter_(1, indices, 1)
        index += 1
        x[:, sum(bn.list_cardinalities[:j]):sum(bn.list_cardinalities[:j+1])] = xx
    return x.to(para['device'])
def load_tarin_dl(data, para, bn):
    x = torch.FloatTensor(len(data[:, 0]), bn.sum_cardinalities).zero_()
    # print(data)
    # print(x)
    # print(BN.list_cardinalities)
    # print(BN.sum_cardinalities)
    # exit(0)

    index = 0
    for j in range(bn.num_nodes): # 0 ~ num_nodes-1
        xx = torch.FloatTensor(len(data[:,0]), bn.list_cardinalities[j]).zero_().scatter_(1, data[:, index].to(int).unsqueeze(1)-1, 1)
        # xx.zero_()
        # mid = data[:, index].to(int).unsqueeze(1)-1
        # xx.scatter_(1, mid, 1)
        index += 1
        x[:, sum(bn.list_cardinalities[:j]):sum(bn.list_cardinalities[:j+1])] = xx
    return x.to(para['device'])