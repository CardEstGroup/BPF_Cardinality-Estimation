from pgmpy.inference import VariableElimination, ApproxInference
import os
from pgmpy.readwrite import BIFReader
import time
# 设置 NUMEXPR_MAX_THREADS 环境变量
os.environ['NUMEXPR_MAX_THREADS'] = '6'

def verify(net_name, data_path):
    # 重命名前后的BN
    reader = BIFReader(data_path + net_name + ".bif")
    net1 = reader.get_model()
    reader = BIFReader(data_path + net_name + "_rename.bif")
    net2 = reader.get_model()

    # 精确推理-VE
    infer1_VE = VariableElimination(net1)
    infer2_VE = VariableElimination(net2)

    # 近似推理
    infer1_AI = ApproxInference(net1)
    infer2_AI = ApproxInference(net2)

    # 执行推理
    t_start = time.time()
    result1_VE = infer1_VE.query(variables=['smoke'], evidence={'lung': 'yes', 'dysp': 'yes'})
    print('原BN的精确推理结果', result1_VE.values)  # [0.91517857 0.08482143]
    t_stop = time.time()
    print("时间: " + str(round((t_stop - t_start), 4))) # 时间: 0.001
    # print(result1_VE)
    # +------------+--------------+
    # | smoke | phi(smoke) |
    # += == == == == == = += == == == == == == =+
    # | smoke(yes) | 0.9152 |
    # +------------+--------------+
    # | smoke(no) | 0.0848 |
    # +------------+--------------+

    t_start = time.time()
    result2_VE = infer2_VE.query(variables=['3'], evidence={'4': '1', '8': '1'})
    print('重命名后BN的精确推理结果', result2_VE.values)  # [0.91517857 0.08482143]
    t_stop = time.time()
    print("时间: " + str(round((t_stop - t_start), 4))) # 时间: 0.001
    # print(result2_VE)
    # +------+----------+
    # | 3    |   phi(3) |
    # +======+==========+
    # | 3(1) |   0.9152 |
    # +------+----------+
    # | 3(2) |   0.0848 |
    # +------+----------+

    t_start = time.time()
    result1_AI = infer1_AI.query(variables=['smoke'], evidence={'lung': 'yes', 'dysp': 'yes'}, n_samples = 10000)
    print()
    print('原BN的近似推理结果', result1_AI.values)  # [0.9179 0.0821]
    t_stop = time.time()
    print("时间: " + str(round((t_stop - t_start), 4))) # 时间: 1.2589

    t_start = time.time()
    result2_AI = infer2_AI.query(variables=['3'], evidence={'4': '1', '8': '1'}, n_samples = 10000)
    print()
    print('重命名后BN的近似推理结果', result2_AI.values)  # [0.9216 0.0784]
    t_stop = time.time()
    print("时间: " + str(round((t_stop - t_start), 4))) # 时间: 0.9887

if __name__ == '__main__':
    bn_name = ['asia']

    for net_name in bn_name:
        data_path = os.path.dirname(os.path.dirname(__file__)) + "\\bn\\" + net_name + "\\"
        print(data_path)
        verify(net_name, data_path)
