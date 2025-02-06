import pandas as pd
from os import path
from torch.utils.data import Dataset


class ML(Dataset):

    def __init__(self, para, BN):
        super(Dataset, self).__init__()
        self.data = self.load(para, BN).values
        # print(self.data)
        # print(self.data.shape) # 维度
        # print(self.data.info()) # 基本信息（维度、列名称、数据格式、所占空间）
        # print(self.data.dtypes) # 数据格式
        # print(self.data.values) # 表的值
        # print(type(self.data.values)) # <class 'numpy.ndarray'>
        # print(self.data.values[0])
        # print(self.data.columns) # 列名称
        # print(self.data.loc[0, 0]) # 第0行
        # print(self.data.loc[0]) # 第0行
        # print(self.data.loc[:, 0].values) # 第0列的值
        # print(len(self.data))
        # print(self.data[0])
        # exit(0)

    def load(self, para, BN):
        data_path = path.dirname(__file__) + '\\'
        # print(data_path) # C:\Users\89647\Desktop\GNN\EBNML\BNML\data\test1
        if para['data_file'] == 'test1':
            return pd.read_table(data_path + r'\test.dat', sep='::', engine='python', header=None)
        else:
            raise NotImplementedError

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, item):
    #     return (self.x1[item], self.x2[item])

