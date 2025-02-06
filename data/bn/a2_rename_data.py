import os
from pgmpy.sampling import BayesianModelSampling
import re
from pgmpy.readwrite import BIFReader
from a1_plot_DAG import plot
#################################### 注意 需要手动再设置一次， 否则出现概率和！=1的错误 ############################################
# init_para.read_cpd_rould_decimal = None
# init_para.read_cpd_rould = None
#################################### 注意， 否则出现概率和！=1的错误 ############################################

def rename_and_generate_data(net_name, data_path):
    print(data_path + net_name + ".bif")
    reader = BIFReader(data_path + net_name + ".bif")
    net = reader.get_model()
    # print(net.nodes) # ['asia', 'tub', 'smoke', 'lung', 'bronc', 'either', 'xray', 'dysp']
    # print(net.edges) # [('asia', 'tub'), ('tub', 'either'), ('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'either'), ('bronc', 'dysp'), ('either', 'xray'), ('either', 'dysp')]
    # print(net.states) # {'asia': ['yes', 'no'], 'smoke': ['yes', 'no'], 'bronc': ['yes', 'no'], 'either': ['yes', 'no'], 'dysp': ['yes', 'no'], 'lung': ['yes', 'no'], 'tub': ['yes', 'no'], 'xray': ['yes', 'no']}
    # print(net.get_cardinality()) #
    # for cpd in net.get_cpds():
    #     print(cpd)
    # exit(0)
    # | asia(yes) | 0.01 |
    # | asia(no) | 0.99 |
    # ...
    # | either | either(yes) | either(no) |
    # | xray(yes) | 0.98 | 0.05 |
    # | xray(no) | 0.02 | 0.95 |

    with open(data_path + net_name + ".bif", 'r', encoding='utf-8') as file:
        file_content = file.read()

    # 1. 将net.node重新赋值为1到n
    nodes_rename = {item: i for i, item in enumerate(net.nodes, start=1)}
    # print(nodes_rename) # {'asia': 1, 'tub': 2, 'smoke': 3, 'lung': 4, 'bronc': 5, 'either': 6, 'xray': 7, 'dysp': 8}
    # 2. 将net.states的key和value重新赋值为1到n
    states_rename = {}
    for i, (key, values) in enumerate(net.states.items(), start=1):
        new_key = i
        new_values = {v: idx + 1 for idx, v in enumerate(values)}
        states_rename[new_key] = new_values
    # print(states_rename) # {1: {'yes': 1, 'no': 2}, 2: {'yes': 1, 'no': 2}, 3: {'yes': 1, 'no': 2}, 4: {'yes': 1, 'no': 2}, 5: {'yes': 1, 'no': 2}, 6: {'yes': 1, 'no': 2}, 7: {'yes': 1, 'no': 2}, 8: {'yes': 1, 'no': 2}}
    # 3. 读取文件
    with open(data_path + net_name + ".bif", 'r', encoding='utf-8') as file:
        file_content = file.read()

    # 4.1 匹配每个块：以network, variable或probability开始  }结束
    blocks = re.findall(r'(network.*?\{.*?\}|variable.*?\{.*?\}|probability.*?\{.*?\})', file_content, re.DOTALL)
    # 4.2 逐块处理
    modified_blocks = []
    for i, block in enumerate(blocks, start=1):
        # 跳过第一个 network 块，不修改
        if i == 1:
            modified_blocks.append(block)
            continue
        # 4.3 修改 variable 块中的 key
        for original_key, new_key in nodes_rename.items():
            block = block.replace(original_key, str(new_key))
        # 4.4 修改 probability 块中的 key 和 value
        for original_key, value_map in states_rename.items():
            for original_value, new_value in value_map.items():
                block = block.replace(original_value, str(new_value))
        # 4.5 将修改后的块添加到列表
        if block.startswith("variable"):
            modified_blocks.append(block + ';\n}')
        else:
            modified_blocks.append(block)
    # 4.6 将修改后的块拼接成完整的文件内容
    modified_content = '\n'.join(modified_blocks)
    # 5. 将修改后的内容保存到新文件
    with open(data_path + net_name + "_rename.bif", 'w', encoding='utf-8') as new_file:
        new_file.write(modified_content)
    # exit(0)


    reader = BIFReader(data_path + net_name + "_rename.bif")
    net = reader.get_model()
    # 前向采样得到数据
    values = BayesianModelSampling(net).forward_sample(int(10000), seed=23)
    # print(type(values))
    # print(values)
    # exit(0)

    # 保存每一行到单独的TXT文件
    lines = []
    for index, row in values.iterrows():
        # 将行转换为逗号分隔的字符串
        line = ','.join(map(str, row.values))
        lines.append(line)
    # 写入文件
    with open(data_path + net_name + "_rename.txt", 'w+', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')  # 每行后添加换行符

    # 画DAG
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\bn\\" + net_name + "\\"
    net_path = data_path + net_name
    plot(net_path + "_rename")


if __name__ == '__main__':
    bn_name = ['asia']

    for net_name in bn_name:
        data_path = os.path.dirname(os.path.dirname(__file__)) + "\\bn\\" + net_name + "\\"
        print(data_path)
        rename_and_generate_data(net_name, data_path)





