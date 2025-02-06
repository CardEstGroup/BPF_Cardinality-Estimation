import torch
import numpy as np
from my_model.mpn import GaussianNPN, vanillaNN
from decimal import Decimal, ROUND_HALF_UP

class BN():
    def __init__(self, num_nodes, list_cardinalities, init_edges, constraint_may_edges, para = None):
        # 检查输入是否正确
        assert (num_nodes > 0)
        assert (len(list_cardinalities) == num_nodes)
        assert (init_edges.shape == (num_nodes, num_nodes))
        assert (constraint_may_edges.shape == (num_nodes, num_nodes))
        # assert (self.findcircle(init_edges) == 0) # 0表示无环，1表示有环

        # BN的各项属性
        self.num_nodes = num_nodes # BN变量个数
        self.list_cardinalities = list_cardinalities # 列表: 变量的势
        self.edges = init_edges # DAG结构
        self.constraint_init_edges = init_edges # 必须存在的边（这里将初始DAG结构作为必须存在边的约束）
        self.constraint_may_edges = constraint_may_edges # 可能存在边的约束
        self.para = para

        self.sum_cardinalities = sum(list_cardinalities)
        self.list_MPN = [None for i in range(0, self.num_nodes)]  # 每个变量对应一个MPN

        self.list_BIC_loglikelihood = [float("-inf") for i in range(0, self.num_nodes)] # 存放每个变量的对数似然
        self.list_BIC_penalty = [float(0) for i in range(0, self.num_nodes)] # 存放每个变量的罚项
        self.list_BIC = [float("-inf") for i in range(0, self.num_nodes)] # 存放每个变量的family_BIC=对数似然-罚项

        # self.list_evidence_node = [] # 列表: 证据变量
        # self.list_search_node = [] # 列表: 查询变量

        # self.list_parent_variables = [self.list_parents(i) for i in range(self.num_nodes)]

    def create_CPD(self, device = "cpu"):
        for i in range(self.num_nodes):
            if self.para['type_MPN'] == 'NPN':
                self.list_MPN[i] = GaussianNPN(self.sum_cardinalities, self.list_cardinalities[i], 'D', self.para['num_neuron_latent_layer'], self.para).to(device)
            elif self.para['type_MPN'] == 'vanillaNN':
                self.list_MPN[i] = vanillaNN(self.sum_cardinalities, self.list_cardinalities[i], 'D', self.para['num_neuron_latent_layer'], self.para).to(device)
            else:
                print('请在初始化文件中将type_MPN设置为MPN或vanillaNN')
                exit(0)
            # print(self.list_MPN[i])
            # print(next(self.list_MPN[i].parameters()).is_cuda)

    def Num_parents_value_combinations(self, i): # 节点i的父节点的取值组合规模
        Num_parents_value_combinations = 1
        for q in self.list_cardinalities * self.edges[:, i]:
            if q > 0:
                Num_parents_value_combinations *= q
        return Num_parents_value_combinations

    def independent_parameters_node_i(self, i): # 节点i的CPT的独立参数个数
        Num_parents_value_combinations = self.Num_parents_value_combinations(i)
        d = Num_parents_value_combinations * (self.list_cardinalities[i]-1)
        return d

    def parameters_node_i(self, i): # 节点i的CPT的独立参数个数
        Num_parents_value_combinations = self.Num_parents_value_combinations(i)
        d = Num_parents_value_combinations * self.list_cardinalities[i]
        return d

    def output_CPD_node(self, i):
        list_parents = self.edges[:, i] # 第i列：节点i的父节点，如 [0 0 0 0 1 1 0 0]
        list_dict_parents_values = []
        # list_parent_cardinalities = self.list_cardinalities * self.edges[:, i] # 节点i父节点的势，如 [0 0 0 0 2 2 0 0]
        # print(list_parent_cardinalities)
        # exit(0)
        Num_parents_value_combinations = self.Num_parents_value_combinations(i) # 节点i父节点的取值组合规模
        # print(Num_parents_value_combinations) # 4
        # exit(0)
        x = torch.FloatTensor(Num_parents_value_combinations, self.sum_cardinalities).zero_().to(self.para['device']) # 为每个节点i父节点的取值组合构建一条MPN的输入
        for index in range(Num_parents_value_combinations):
            temp = index # 第0~3个取值组合
            dict_parents_values = {} # 创建用于存储{父节点1：父节点1取值,...,父节点k：父节点k取值}的字典
            for j in range(self.num_nodes): # j = 0, 1, 2, 3, 4, 5, 6, 7
                reverse_j = (self.num_nodes - 1) - j # reverse_j= 7, 6, 5, 4, 3, 2, 1, 0，即优先遍历排序靠后的父节点取值
                # print('reverse_j: ' + str(reverse_j))
                if self.edges[reverse_j][i] == 1: # 1个父节点
                    q = torch.tensor([temp % self.list_cardinalities[reverse_j] + 1]) # 该父节点的1个取值
                    dict_parents_values[reverse_j+1] = q.item() # 向字典中添加键值对——实际父节点的序号:父节点取值
                    # print(temp)
                    # print(self.list_cardinalities[reverse_j])
                    # print('q: ' + str(q))
                    temp = (temp / (self.list_cardinalities[reverse_j])).__int__() # 用于求下一排序的父节点取值
                    # print(temp)
                    # print(torch.FloatTensor(1, self.list_cardinalities[reverse_j]).zero_())
                    xx = torch.FloatTensor(1, self.list_cardinalities[reverse_j]).zero_().scatter_(1, q.to(int).unsqueeze(1)-1, 1) # 该父节点取值的one-hot
                    # print(xx)
                    x[index][sum(self.list_cardinalities[:reverse_j]):sum(self.list_cardinalities[:reverse_j+1])] = xx # 将该父节点取值的one-hot覆盖到第index行的相应列
                    # print(x)
            # list_dict_parents_values.append(dict_parents_values) # [{6: 1, 5: 1}, {6: 2, 5: 1}, {6: 1, 5: 2}, {6: 2, 5: 2}]
            list_dict_parents_values.append(dict(sorted(dict_parents_values.items()))) # [{5: 1, 6: 1}, {5: 1, 6: 2}, {5: 2, 6: 1}, {5: 2, 6: 2}]
        # print(x)
        # exit(0)
        #    r_j=0   r_j=1
        #    q  t    q  t
        # 0: 1  0    1  0
        # 1: 2  0    1  0
        # 2: 1  1    2  0
        # 3: 2  1    2  0

        data_s = torch.zeros(x.size()).to(self.para['device']) # MPN输入的方差，恒为0，可忽视
        # print(data_s)
        # exit(0)
        output = self.list_MPN[i]((x, data_s))[0]
        # print('---------CPD of Node: {}-----------'.format(i+1))
        # print(output[0])

        # print(output.tolist())
        # print(list_dict_parents_values)
        return output, list_dict_parents_values

    def list_parents(self, i):
        list_parents = []
        for j in range(self.num_nodes):  # 0 ~ num_nodes-1
            if self.edges[j][i] == 1:  # 父节点
                list_parents.append(j+1)
        return list_parents

    def adjust_values_to_sum_one(self, values):
        # 保留8位小数
        values = [Decimal(f"{v:.8f}") for v in values]
        # 计算当前和
        total_sum = sum(values)
        # 确定最大值的索引
        max_index = values.index(max(values))
        # 计算剩余值的和
        remaining_sum = total_sum - values[max_index]
        # 重新计算最大值，使得和为1
        values[max_index] = Decimal('1') - remaining_sum
        # 将所有值四舍五入到8位小数
        values = [v.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP) for v in values]
        # 最终验证总和是否为1
        if sum(values) != 1:
            print('输出pgmpy的bif文件时，CPT某行的概率处理后和不为1！')
            exit(0)
        return values
    def my_output_bif_for_pgmpy(self, file_path, node_names):
        list_blocks = []
        # 1. 向blocks中插入1个 network块
        list_blocks.append('network unknown {\n}')
        # 2. 向blocks中插入多个 variable块
        for i in range(self.num_nodes):
            block = 'variable '
            # block += str(i+1)
            block += node_names[i]
            block += ' {\n  type discrete [ '
            block += str(self.list_cardinalities[i])
            block += ' ] { '
            block += ', '.join(map(str, [c for c in range(1, self.list_cardinalities[i]+1)]))
            block += ' };\n}'
            # print(block)
            list_blocks.append(block)
        # 3. 向blocks中插入多个 probability块
        for i in range(self.num_nodes):
            block = 'probability ( ' # ‘probability ( ‘
            # block += str(i + 1) # ’probability ( 8’
            block += node_names[i] # ’probability ( 8’
            if len(self.list_parents(i)) > 0:
                block += ' | ' #  ‘probability ( 8 | ’
                # block += ', '.join(map(str, self.list_parents(i))) # ‘probability ( 8 | 5, 6’
                block += ', '.join([node_names[index-1] for index in self.list_parents(i)]) # ‘probability ( 8 | 5, 6’
            block += ' ) {\n' # ’probability ( 8 | 5, 6 ) {‘

            output, list_dict_parents_values = self.output_CPD_node(i)
            for dict_parents_values, probilities in zip(list_dict_parents_values, output.tolist()):
                # print(dict_parents_values) # {5: 1, 6: 1}
                # print(probilities)# [0.91412736, 0.08587264]
                if len(self.list_parents(i)) > 0:
                    block += '  ('
                    block += ', '.join(map(str, list(dict_parents_values.values())))
                    block += ') '
                else:
                    block += '  table '
                adjust_probilities = self.adjust_values_to_sum_one(probilities)
                block += ', '.join(map(str, adjust_probilities))
                block += ';\n'
            block += '}'
            list_blocks.append(block)
        # 4. 将修改后的块拼接成完整的文件内容
        modified_content = '\n'.join(list_blocks)
        # 5。 将修改后的内容保存到新文件
        with open(file_path + ".bif", 'w', encoding='utf-8') as new_file:
            new_file.write(modified_content)

    def output_bif_for_pgmpy(self, file_path):
        list_blocks = []
        # 1. 向blocks中插入1个 network块
        list_blocks.append('network unknown {\n}')
        # 2. 向blocks中插入多个 variable块
        for i in range(self.num_nodes):
            block = 'variable '
            block += str(i+1)
            block += ' {\n  type discrete [ '
            block += str(self.list_cardinalities[i])
            block += ' ] { '
            block += ', '.join(map(str, [c for c in range(1, self.list_cardinalities[i]+1)]))
            block += ' };\n}'
            # print(block)
            list_blocks.append(block)
        # 3. 向blocks中插入多个 probability块
        for i in range(self.num_nodes):
            block = 'probability ( ' # ‘probability ( ‘
            block += str(i + 1) # ’probability ( 8’
            if len(self.list_parents(i)) > 0:
                block += ' | ' #  ‘probability ( 8 | ’
                block += ', '.join(map(str, self.list_parents(i))) # ‘probability ( 8 | 5, 6’
            block += ' ) {\n' # ’probability ( 8 | 5, 6 ) {‘

            output, list_dict_parents_values = self.output_CPD_node(i)
            for dict_parents_values, probilities in zip(list_dict_parents_values, output.tolist()):
                # print(dict_parents_values) # {5: 1, 6: 1}
                # print(probilities)# [0.91412736, 0.08587264]
                if len(self.list_parents(i)) > 0:
                    block += '  ('
                    block += ', '.join(map(str, list(dict_parents_values.values())))
                    block += ') '
                else:
                    block += '  table '
                adjust_probilities = self.adjust_values_to_sum_one(probilities)
                block += ', '.join(map(str, adjust_probilities))
                block += ';\n'
            block += '}'
            list_blocks.append(block)
        # 4. 将修改后的块拼接成完整的文件内容
        modified_content = '\n'.join(list_blocks)
        # 5。 将修改后的内容保存到新文件
        with open(file_path + ".bif", 'w', encoding='utf-8') as new_file:
            new_file.write(modified_content)


    # def copy_CPD(self, i = None, BN = None):
    #     assert(BN != None)
    #     if i == None:
    #         for i in range(self.num_nodes): # 0 ~ num_nodes-1
    #             self.list_MPN[i] = BN.list_MPN[i].copy()
    #     else:
    #         # print(type(i))..
    #         assert(type(i) == int)
    #         print(BN.list_MPN[i])
    #         print(BN.list_MPN[i].copy())
    #         exit(0)
    #         self.list_MPN[i] = BN.list_MPN[i].copy()



    def copy_BN_without_CPD(self):
        return BN(self.num_nodes, self.list_cardinalities, self.edges, self.constraint_may_edges, self.para)

    def dfs(self, G, i, color): # 用于后续的findcircle方法中
        r = len(G)
        color[i] = -1
        have_circle = 0
        for j in range(r):	# 遍历当前节点i的所有邻居节点
            if G[i][j] != 0:
                if color[j] == -1:  # 如果遇到一个正在被访问的节点，说明有环
                    have_circle = 1
                elif color[j] == 0: # 如果是未访问的节点，递归调用DFS
                    have_circle = self.dfs(G, j, color) or have_circle  # 保持之前的have_circle状态
        color[i] = 1
        return have_circle

    def findcircle(self, G): # 判断DAG是否满足无环
        # color = 0 该节点暂未访问
        # color = -1 该节点访问了一次
        # color = 1 该节点的所有孩子节点都已访问,就不会再对它做DFS了
        r = len(G)
        color = [0] * r
        have_circle = 0
        for i in range(r):	# 遍历所有的节点
            if color[i] == 0:
                have_circle = self.dfs(G, i, color) or have_circle  # 保持之前的have_circle状态
        return have_circle

    # def list_parent_nodes(self, i):
    #     list_parent_nodes = []
    #     for j in range(self.num_nodes):  # 0 ~ num_nodes-1
    #         if self.edges[j][i] == 1:  # 父节点
    #             list_parent_nodes.append(j)
    #     return list_parent_nodes
    #
    # def list_child_nodes(self, i):
    #     list_child_nodes = []
    #     for j in range(self.num_nodes):  # 0 ~ num_nodes-1
    #         if self.edges[i][j] == 1:  # 父节点
    #             list_child_nodes.append(j)
    #     return list_child_nodes





if __name__ == '__main__':

    num_nodes = 8 # BN变量个数
    list_cardinalities = [2 for i in range(8)] # 列表: 1表示连续变量的势，1~n表示离散变量的势
    list_edges = [[1, 2], [1, 3]] # 列表: 初始边的集合

    init_edges = np.zeros((num_nodes, num_nodes), dtype = int, order='C') # 初始边的邻接矩阵
    for e in list_edges:
        init_edges[e[0]-1][e[1]-1] = 1
    constraint_may_edges = init_edges
    # print(constraint_may_edges)
    # print(constraint_may_edges[0,:])
    # print(constraint_may_edges[:,1]) #
    # exit(0)


    BN = BN(num_nodes, list_cardinalities, init_edges, constraint_may_edges)
    print(BN.edges)
    print(BN.findcircle(BN.edges))
    exit(0)




