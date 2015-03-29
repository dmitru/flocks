__author__ = 'dmitru'

import networkx as nx

class ComGraphUtils:
    '''Contains functions for generating different communication graphs'''
    @staticmethod
    def arrow_tip_4():
        G = nx.DiGraph()
        for v in range(4):
            G.add_node(v)
        G.add_edge(2, 0)
        G.add_edge(1, 0)
        G.add_edge(1, 2)
        G.add_edge(3, 2)
        G.add_edge(3, 0)
        return G

    @staticmethod
    def full_graph(n):
        G = nx.DiGraph()
        for v in range(n):
            G.add_node(v)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                G.add_edge(i, j)
        return G

    @staticmethod
    def arrow_tip_6():
        G = nx.DiGraph()
        for v in range(6):
            G.add_node(v)
        G.add_edge(1, 0)
        G.add_edge(2, 0)
        G.add_edge(3, 1)
        G.add_edge(3, 2)
        G.add_edge(4, 3)
        G.add_edge(5, 3)
        G.add_edge(4, 1)
        G.add_edge(5, 2)
        return G