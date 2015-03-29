__author__ = 'dmitru'

import numpy as np

def position_vector(vpv):
    return np.kron(vpv[0::4], np.array([1,0])) + np.kron(vpv[2::4], np.array([0,1]))

def velocity_vector(vpv):
    return np.kron(vpv[1::4], np.array([1,0])) + np.kron(vpv[3::4], np.array([0,1]))

class FormationsUtil:
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