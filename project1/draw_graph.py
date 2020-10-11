import networkx as nx
import pandas as pd
import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    raise Exception("usage: python draw_graph.py <data_file>.csv <edge_file>.gph <out_file>.png")

data_file = sys.argv[1]
edge_file = sys.argv[2]
out_file = sys.argv[3]

D = pd.read_csv(data_file)
variables = list(D.columns)
G = nx.DiGraph()
G.add_nodes_from(variables)

with open(edge_file, 'r') as f:
    for line in f:
        e = line.strip().split(",")
        G.add_edge(*e)


pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color="w", alpha=0.4)
nx.draw_networkx_edges(G, pos, alpha=0.4, node_size=0, width=1, edge_color="b")
nx.draw_networkx_labels(G, pos, font_size=8)
plt.savefig(out_file)
