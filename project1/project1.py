import sys
import pandas as pd
import numpy as np
from scipy.special import gamma, gammaln
import math
from collections import defaultdict
import itertools

import networkx as nx

counts = None
variables = None
variable_values = dict()
D = None

def write_gph(dag, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{},{}\n".format(edge[0], edge[1]))


def idx2dto1d(row, var_name, parents):
    if len(parents) == 0:
        return 0

    j = []
    shape = []
    for var_name_ in variables:
        if var_name_ != var_name and var_name_ in parents:
            j.append(row[var_name_]-1)
            shape.append(variable_values[var_name_])
    return np.ravel_multi_index(j, tuple(shape))


def populate_counts(G):
    # because gamma(1)/gamma(1) = 1, and log 1 = 0, 
    # we don't need to care about the "missing" instantiations because they amount to 0 in the bayesian score
    global counts

    n = len(variables)
    counts = dict()
    for index, row in D.iterrows():
        for i in range(n):
            i = i   
            var_name = variables[i]
            parents = [var_name_ for var_name_ in G.predecessors(var_name)]
            j = idx2dto1d(row, var_name, parents)
            k = row[var_name]
            if i not in counts:
                counts[i] = dict()
            if j not in counts[i]:
                counts[i][j] = defaultdict(int)
            counts[i][j][k] += 1
    #return counts


def update_counts(G, var_name):
    # do dynamic programming to propagate the change in parents?
    global counts

    n = len(variables)
    parents = G.predecessors(var_name)
    i = variables.index(var_name)
    counts[i] = dict()
    for index, row in D.iterrows():
        parents = [var_name_ for var_name_ in G.predecessors(var_name)]
        j = idx2dto1d(row, var_name, parents)
        k = row[var_name]
        if j not in counts[i]:
            counts[i][j] = defaultdict(int)
        counts[i][j][k] += 1


def bayesian_score(G):
    # drop log P(G)
    n = len(variables)
    res = 0
    for i in range(n):
        component = 0
        for j in counts[i]:
            # all pseudo counts are 1
            alpha_sum = variable_values[variables[i]]
            m_sum = np.sum([counts[i][j][k] for k in counts[i][j]])
            component += gammaln(alpha_sum) - gammaln(alpha_sum + m_sum)
            for k in counts[i][j]:
                alpha_ijk = 1 
                m_ijk = counts[i][j][k]
                component += gammaln(alpha_ijk + m_ijk) - gammaln(alpha_ijk)
        res += component
    return res


def random_directed_graph(p=0.2):
    # generate arbitrary ordering of nodes
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            if np.random.uniform() < p:
                G.add_edge(variables[i], variables[j])
    return G

    """
    # limit number of parents for a given node
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    edges = itertools.permutations(variables, 2)
    for e in edges:
        if np.random.uniform() < p:
            G.add_edge(*e)
    return G
    """


def rand_graph_neighbor(G):
    nodes = list(G.nodes)
    edges = list(G.edges)
    i = np.random.randint(len(nodes))
    j = (i + np.random.randint(1, len(nodes))) % len(nodes)
    G_ = G.copy()
    if (nodes[i], nodes[j]) in edges:
        G_.remove_edge(nodes[i], nodes[j])
        update_counts(G_, nodes[j])
    else:
        G_.add_edge(nodes[i], nodes[j])
        update_counts(G_, nodes[j])
    return G_


def is_cyclic(G):
    try: 
        nx.find_cycle(G, orientation='original')
    except nx.exception.NetworkXNoCycle:
        return False
    
    return True


def hill_climbing(D, outfile, k_max=20):
    """
    Returns a graph instantiated by greedy local search algorithm.
    """
    G = random_directed_graph()
    while is_cyclic(G):
        G = random_directed_graph()
    print("writing graph")
    write_gph(G, outfile)
    populate_counts(G)
    y = bayesian_score(G)
    for k in range(k_max):
        G_ = rand_graph_neighbor(G)
        if is_cyclic(G_):
            y_ = float('-inf')
            print('cyclic')
        else:
            y_ = bayesian_score(G_)
        if y_ > y:
            y, G = y_, G_
            print("writing improved graph")
            write_gph(G, outfile)

    return G


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    global D, variables, variable_values
    D = pd.read_csv(infile)
    variables = list(D.columns)
    for var in variables:
        num_values = D[var].max()
        variable_values[var] = num_values
    
    # implement simple algorithm
    G = hill_climbing(D, outfile)

    # convert to file
    write_gph(G, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
