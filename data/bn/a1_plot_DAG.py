import networkx as nx
from pgmpy.readwrite import BIFReader
from graphviz import Digraph
import os

def plot(net_path):

    # print(data_path)
    # exit(0)

    reader = BIFReader(net_path + ".bif", n_jobs=1)
    network = reader.get_model()

    G = Digraph('network')
    # nodes = network.nodes()
    # print(nodes)
    # exit(0)

    # nodes_sort = list(nx.topological_sort(network))
    # print('nodes_num:', len(nodes_sort))
    # print('nodes_sort:', nodes_sort)
    # exit(0)

    edges = network.edges()
    # print('\nedges_num:', len(edges))
    # print('\nedges:', edges)
    # exit(0)

    for a, b in edges:
        G.edge(a, b)
    # var_card = {node: network.get_cardinality(node) for node in nodes_sort}
    # print('var_card:', var_card)
    # var_card=dict(zip(cpd.variables, cpd.cardinality))

    G.render(net_path + ".gv", view=False)
    # G
    # print(len(nodes))


if __name__ == '__main__':
    bn_name = ['asia']
    for net_name in bn_name:
        data_path = os.path.dirname(os.path.dirname(__file__)) + "\\bn\\" + net_name + "\\"
        net_path = data_path + net_name
        plot(net_path)