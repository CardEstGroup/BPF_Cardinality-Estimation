import pandas as pd
from os import path
from torch.utils.data import Dataset


class ML(Dataset):

    def __init__(self, para):
        super(Dataset, self).__init__()
        d = self.load(para)
        self.x1 =d['x1']
        self.x2 =d['x2']
        self.w = d['w']
        self.prior1 = d['prior1']
        self.prior2 = d['prior2']

    def load(self, para):
        data_path = path.dirname(__file__) + '\\'
        # print(data_path) # C:\Users\89647\Desktop\GNN\EBNML\BNML\data\test1
        if para['data_file'] == 'test1':
            return pd.read_table(data_path + r'\test.dat', sep='::', names=['x1', 'x2', 'w', 'prior1', 'prior2'], engine='python')
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, item):
        return (self.x1[item], self.x2[item], self.w[item], self.prior1[item], self.prior2[item])

