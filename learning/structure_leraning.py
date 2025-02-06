import numpy as np
import copy
import time
from learning.CPD_learning_process import CPD_train_multi_batch, CPD_train_no_batch

def generate_candidate_DAG_HC(BN, num_k2_edges = 100):
    list_candidate_DAG = []
    list_candidate_changed_node = [] # 存放每个候选模型中父节点发生变化的节点：用于后续仅计算父节点发生变化的节点的FBIC，进而得到候选模型的BIC评分，可避免计算所有节点的FBIC，提升效率
    list_candidate_changed_egde = [] # 存放每个候选模型增加的边
    # print("edges:")
    # print(BN.edges)
    # print("constraint_init_edges:")
    # print(BN.constraint_init_edges)
    # print("constraint_may_edges:")
    # print(BN.constraint_may_edges)
    # exit(0)

    # 加边：对两两节点之间逐一加边
    candidate_DAG = None
    for i in range(0, BN.num_nodes):
        for j in range(0, BN.num_nodes):
            # print(i, j)
            # 节点自身、或节点i和节点j之间已有边，则不加边
            if BN.constraint_may_edges[i][j] == 0 or i == j or BN.edges[i][j] == 1 or sum(BN.edges[:, j]) >= num_k2_edges:
                # print("节点自身，或者i和j之间已有边则跳过")
                continue
            else:
                candidate_DAG = BN.edges.copy() # A是B的深复制：将当前BN结构复制给临时变量candidate_DAG
                candidate_DAG[i][j] = 1 # 加边
                # print(candidate_DAG)
                # exit(0)
                if BN.findcircle(candidate_DAG) == 0: # 0表示无环，1表示有环
                    list_candidate_DAG.append(candidate_DAG)
                    list_candidate_changed_node.append(j)
                    list_candidate_changed_egde.append([i, j])
                    # print(list_candidate_DAG)
    return list_candidate_DAG, list_candidate_changed_node, list_candidate_changed_egde


def structure_train(BN, para, data):

    epoch = 0
    while 1:
        epoch += 1 # 结构学习迭代次数
        t_start = time.time()
        # 生成候选模型
        list_candidate_DAG, list_candidate_changed_node, list_candidate_changed_egde = generate_candidate_DAG_HC(BN)
        # print("num_candidate_DAG:" + str(len(list_candidate_DAG)))
        # print("list_candidate_DAG:")
        # print(list_candidate_DAG)
        # print("list_candidate_changed_node:")
        # print(list_candidate_changed_node)
        # print("list_candidate_changed_egde:")
        # print(list_candidate_changed_egde)
        # exit(0)
        if list_candidate_DAG == []:
            print('No candidate!')

        current_optimal_model = None # 当前最优模型
        current_optimal_BIC_increase = float(0) # 当前最优模型的BIC增加值

        # （1）利用“训练数据”，更新每个候选模型中父节点发生变化的节点的MPN，得到该节点“新的CPT”
        # （2）基于“训练数据“和”新CPT”得到likelihood，该节点的FBIC=likelihood-惩罚项，进而得到候选模型的BIC评分
        for index in range(len(list_candidate_DAG)):
            # 第i个节点
            i = list_candidate_changed_node[index]
            egde = list_candidate_changed_egde[index]
            candidate_DAG = list_candidate_DAG[index]
            print()
            print('-------Structure Learning Epoch: {}, 第 {} 个候选模型, 第 {} 个节点存在加边 <{}, {}>-------'.format(epoch, index+1, i+1, egde[0]+1, egde[1]+1))
            # print(candidate_DAG)
            # print(BN)
            # candidate_model = BN
            candidate_model = BN.copy_BN_without_CPD()
            # candidate_model = copy.deepcopy(BN)
            # print(BN)
            # print(candidate_model)
            # exit(0)

            candidate_model.edges = candidate_DAG
            # print(BN.edges)
            # print(candidate_model.edges)
            # exit(0)
            candidate_model.list_MPN[i] = copy.deepcopy(BN.list_MPN[i])
            # print(BN.list_MPN)
            # print(candidate_model.list_MPN)
            # exit(0)

            candidate_model.list_BIC_loglikelihood[i] = CPD_train_no_batch(candidate_model, i, para, data)
            candidate_model.list_BIC_penalty[i] = candidate_model.independent_parameters_node_i(i) / 2 * np.log(para['train_size'])
            candidate_model.list_BIC[i] = candidate_model.list_BIC_loglikelihood[i] - candidate_model.list_BIC_penalty[i]
            print('list_BIC_loglikelihood:' + str(candidate_model.list_BIC_loglikelihood[i]))
            print('list_BIC_penalty:' + str(candidate_model.list_BIC_penalty[i]))
            print('list_BIC:' + str(candidate_model.list_BIC[i]))
            print(candidate_model.output_CPD_node(i)[0])  # 输出CPT
            # exit(0)

            # print(candidate_model.list_BIC)
            # print(BN.list_BIC)
            candidate_model_BIC_increased = candidate_model.list_BIC[i] - BN.list_BIC[i]
            print('candidate_model_BIC_increased:' + str(candidate_model_BIC_increased))
            # exit(0)
            if candidate_model_BIC_increased > current_optimal_BIC_increase: # 将当前模型作为最优模型
                current_optimal_BIC_increase = candidate_model_BIC_increased
                # print('current_optimal_BIC_increase:')
                # print(current_optimal_BIC_increase)
                current_optimal_model = copy.deepcopy(candidate_model)
                # print(current_optimal_model)
            del candidate_model
            # print(current_optimal_model)
            # print(current_optimal_model.list_BIC_loglikelihood[i])
            # print(current_optimal_model.list_BIC_penalty[i])
            # print(current_optimal_model.list_BIC[i])
            # print(current_optimal_model.list_MPN)

        # exit(0)
        t_stop = time.time()
        print("一次结构学习的时间: " + str(round((t_stop - t_start), 4)))
        print('Train Epoch: {}, current_optimal_BIC_increase: {:.4f}'.format(epoch, current_optimal_BIC_increase))

        if current_optimal_BIC_increase == 0: # 收敛，返回结果
            return BN
        else: # 将当前最优模型覆盖到BN，并继续迭代
            BN.edges = current_optimal_model.edges
            print('Current optimal model:')
            print(BN.edges)
            # print(current_optimal_model.list_MPN)
            for i in range(0, BN.num_nodes):
                if current_optimal_model.list_MPN[i] != None:
                    BN.list_MPN[i] = copy.deepcopy(current_optimal_model.list_MPN[i])
                    BN.list_BIC_loglikelihood[i] = current_optimal_model.list_BIC_loglikelihood[i]
                    BN.list_BIC_penalty[i] = current_optimal_model.list_BIC_penalty[i]
                    BN.list_BIC[i] = current_optimal_model.list_BIC[i]
        del current_optimal_model
        # print(BN.edges)
        # print(BN.list_MPN)
        # exit(0)

        if epoch == para['structure_learning_max_iter_num']:
            print('Structure learning epoch: {}, and epoch has been equal to convergence_iter_num: {}'.format(epoch, para['structure_learning_max_iter_num']))
            return BN

