from datetime import datetime
import os

__author__ = 'dmitru'

import numpy as np
import random

def position_vector(vpv):
    return np.kron(vpv[0::4], np.array([1,0])) + np.kron(vpv[2::4], np.array([0,1]))

def velocity_vector(vpv):
    return np.kron(vpv[1::4], np.array([1,0])) + np.kron(vpv[3::4], np.array([0,1]))

class FormationsUtil:
    @staticmethod
    def rotate_90c(h):
        res = np.zeros(len(h))
        for i in range(int(len(h) / 2)):
            res[2*i] = h[2*i + 1]
            res[2*i + 1] = -h[2*i]
        return res

    @staticmethod
    def arrow_tip_4():
        '''Returns vector, describing formation of arrow tip configuration for 4 agents'''
        return np.array([0.0, 0.0,
                         1.0, 1.5,
                         1.0, 3.0,
                         2.0, 0.0])
    @staticmethod
    def arrow_tip_6():
        '''Returns vector, describing formation of arrow tip configuration for 6 agents'''
        return np.array([0.0, 0.0,
                         1.0, 0.0,
                         2.0, 0.0,
                         0.5, 1.5,
                         1.5, 1.5,
                         1.0, 3.0])

    @staticmethod
    def square_4():
        return np.array([0.0, 0.0,
                        0.0, 1.0,
                        1.0, 1.0,
                        1.0, 0.0])

    @staticmethod
    def line_y(k):
        t = []
        for i in range(k):
            t.append(0.0)
            t.append(float(i))
        return np.array(t)

    @staticmethod
    def line_x(k):
        t = []
        for i in range(k):
            t.append(float(i))
            t.append(0.0)
        return np.array(t)

    @staticmethod
    def compute_closeness(h, x):
        '''Computes how close the given configuration x in XV-space to desired formation h'''
        ones = np.ones(h.size / 4)
        es = [np.kron(ones, np.array([1,0,0,0])),
              np.kron(ones, np.array([0,1,0,0])),
              np.kron(ones, np.array([0,0,1,0])),
              np.kron(ones, np.array([0,0,0,1]))]
        x_relative = np.array(x)
        h_relative = np.array(h)
        for i in range(4):
            x_relative -= x[i] * es[i]
            h_relative -= h[i] * es[i]
        delta = x_relative - h_relative
        return np.linalg.norm(delta)

    @staticmethod
    def random_positions(num_agents, r=1.0):
        result = [2 * r * (random.random() - 0.5) for _ in range(2*num_agents)]
        return result

    @staticmethod
    def extend_position_to_vpv(pos, vx0=0, vy0=0):
        assert len(pos) % 2 == 0
        n = len(pos) / 2
        ones = np.ones(n)
        return np.kron(np.array(pos), np.array([1, 0])) + \
            vx0 * np.kron(ones, np.array([0, 1, 0, 0])) + \
            vy0 * np.kron(ones, np.array([0, 0, 0, 1]))

def laplace_matrix(G):
    import networkx as nx
    Q = nx.linalg.adjacency_matrix(G).todense()
    Diag = np.diag([sum([1 for e in G.edges() if e[0] == i]) for i in G.nodes()])
    return (Diag - Q)

def rot_matrix(v):
        '''Returns the two-dimensional rotation matrix 2x2 that
        rotates vector e1 to point in v direction'''
        #print(np.linalg.norm(v))
        t = np.array(v) / np.linalg.norm(v)
        return np.array([
            [t[0], -t[1]],
            [t[1], t[0]]
        ])

import matplotlib.cm as cmx
import matplotlib.colors as colors

class ExperimentManager:
    @staticmethod
    def init(name='experiment', root_path='.', append_timestamp=True):
        np.random.seed(123)

        ExperimentManager.counter = 0
        ExperimentManager.name = name
        if append_timestamp:
            now = datetime.now()
            ExperimentManager.name += '_' + '%d_%d_%d__%d_%d_%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        ExperimentManager.base_name = os.path.join(root_path, name)
        if not os.path.exists(ExperimentManager.base_name):
            os.makedirs(ExperimentManager.base_name)

    @staticmethod
    def next_filename(suffix='', extension='.png', increment=True):
        res = os.path.join(ExperimentManager.base_name, '%s_%s_%d%s' % (ExperimentManager.name, suffix, ExperimentManager.counter, extension))
        if increment:
            ExperimentManager.counter += 1
        return res

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

if __name__ == '__main__':
    for i in range(1, 8):
        print(FormationsUtil.random_positions(i))

    for i in range(1, 8):
        p = FormationsUtil.random_positions(i)
        print(FormationsUtil.extend_position_to_vpv(p, vx0=1.0, vy0=-1.0))