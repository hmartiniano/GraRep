import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from texttable import Texttable

def create_inverse_degree_matrix(edges):
    """
    Creating an inverse degree matrix from an edge list.
    :param edges: Edge list.
    :return D_1: Inverse degree matrix.
    """
    graph = nx.from_edgelist(edges)
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in range(graph.number_of_nodes())]
    D_1 = sparse.coo_matrix((degs,(ind,ind)),shape=(graph.number_of_nodes(), graph.number_of_nodes()),dtype=np.float32)
    return D_1

def normalize_adjacency(graph):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param edges: Edge list of graph.
    :return A: Normalized adjacency matrix.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    D_1 = sparse.coo_matrix((degs,(ind,ind)),shape=(graph.number_of_nodes(), graph.number_of_nodes()),dtype=np.float32)
    A = nx.adjacency_matrix(graph).dot(D_1)
    return A

def read_graph(edge_path):
    """
    Method to read graph and create a target matrix.
    :param edge_path: Path to the ege list.
    :return A: Target matrix.
    """
    edges = pd.read_csv(edge_path)
    graph = nx.from_pandas_edgelist(edges, source="id_1", target="id_2", edge_attr="weight")
    A = normalize_adjacency(graph)
    return A, graph.nodes()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.items()])
    print(t.draw())
