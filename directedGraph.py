import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
# from plotGraph import plotGraph

'''
BitcoinAlpha has 3783 nodes, 24186 edges
'''

def snapDataset():
    fh = open("soc-sign-bitcoinalpha.csv", 'rb')
    G = nx.read_weighted_edgelist(fh, create_using=nx.DiGraph(), nodetype=int, delimiter=",")
    fh.close()
    return G

def plotDirectedGraph(G):
    # labels = nx.get_edge_attributes(G, 'weight')
    pos = nx.spring_layout(G)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw(G, pos, with_labels=True)
    plt.show()

def interactiveDirectedGraph(G):
    # plot simple interactive graph using pyvis
    nt = Network("500px", "1000px", directed=True)
    nt.from_nx(G)
    nt.show("nx.html")

def createAdjacencyMatrix(G):
    adj = nx.convert_matrix.to_numpy_matrix(G)
    # adj = nx.convert_matrix.to_pandas_adjacency(G)
    # adj = nx.convert_matrix.to_pandas_edgelist(G)
    return adj

G = snapDataset()
print(createAdjacencyMatrix(G).shape)
# print(sorted(G.in_degree, key=lambda x: x[1], reverse=True))
# print(sorted(G.out_degree, key=lambda x: x[1], reverse=True))
# print(G.get_edge_data(2,7500))
# print(G.out_degree(7500))