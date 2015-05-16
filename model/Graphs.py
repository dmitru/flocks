__author__ = 'dmitru'

import networkx as nx
import random
from random import randint

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

    @staticmethod
    def random_graph(v, e, directed=True):
        '''Returns a random graph containing spanning tree with v vertices and e edges'''
        assert (e <= (v*(v-1))/2 and not directed) or (e <= (v*(v-1)) and directed)
        G = nx.DiGraph()
        for i in range(v):
            G.add_node(i)
        in_spanning_tree = [0]
        edges_counter = 0
        for i in range(1, v):
            j = random.sample(in_spanning_tree, 1)[0]
            in_spanning_tree.append(i)
            G.add_edge(j, i)
            if directed:
                G.add_edge(i, j)
            edges_counter += 1
        while edges_counter < e:
            a, b = random.randint(1, v - 1), random.randint(1, v - 1)
            if a == b:
                continue
            if G.has_edge(a, b):
                continue
            edges_counter += 1
            G.add_edge(a, b)
            if directed:
                G.add_edge(b, a)
        return G



