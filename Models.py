__author__ = 'dmitru'

import scipy
import scipy.integrate
import networkx as nx
import numpy as np

import scipy.integrate as spi

from Utils import velocity_vector, position_vector

class LinearModel:
    '''Describes two-dimensional model of moving agents with communication graph'''
    def __init__(self, com_graph, num_agents, A, B, F, h, x0):
        '''Parameters:
        com_graph - communication graph that agents use,
        agents - list of Agent objects,
        A - 4x4 matrix, B - 4x2 matrix, see [1].II for details
        F - 2x4 feedback matrix
        h - desired formation
        x0 - initial position-velocity vector of all agents
        '''
        self.N = num_agents
        assert(type(com_graph) == nx.DiGraph)
        assert(len(com_graph.nodes()) == self.N)

        assert(A.shape == (4, 4))
        assert(B.shape == (4, 2))
        assert(F.shape == (2, 4))
        assert(h.size == num_agents * 4)
        assert(x0.size == num_agents * 4)

        self.CG = com_graph
        Q = nx.linalg.adjacency_matrix(com_graph).todense()
        D = np.diag([sum([1 for e in com_graph.edges() if e[0] == i]) for i in com_graph.nodes()])
        self.LG = np.dot(np.linalg.pinv(D), (D - Q))
        self.A = A
        self.B = B
        self.F = F
        self.x0 = x0
        self.h = h
        self.k = A[3,1]
        self.K = np.dot(self.B, self.F)
        self.T1 = np.kron(self.LG, self.K)

    def compute(self, T, dt):
        def dx_dt(x, t=0):
            y = np.dot(np.kron(np.eye(self.N), self.A), x) + np.dot(self.T1, (x - self.h))
            return np.asarray(y)[0,:]

        ts = scipy.linspace(0, T, num=T/dt)
        xys = scipy.integrate.odeint(dx_dt, self.x0, ts)
        return xys

    @staticmethod
    def circular_from_com_graph(com_graph, h, x0, k, f1, f2):
        n = len(com_graph.nodes())
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, -k],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, k,   0.0, 0.0]])
        B = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 1.0]])
        F = np.array([[f1,  f2,  0.0, 0.0],
                      [0.0, 0.0, f1,  f2]])
        return LinearModel(com_graph, n, A, B, F, h, x0)

class OrientableModel:
    '''Describes two-dimensional model of moving agents with communication graph,
    formation reorients intself along the line of the flight'''
    def __init__(self, com_graph, num_agents, A, B, F, h, x0):
        '''Parameters:
        com_graph - communication graph that agents use,
        agents - list of Agent objects,
        A - 4x4 matrix, B - 4x2 matrix, see [1].II for details
        F - 2x4 feedback matrix
        h - desired formation
        x0 - initial position-velocity vector of all agents
        '''
        self.N = num_agents
        assert(type(com_graph) == nx.DiGraph)
        assert(len(com_graph.nodes()) == self.N)

        assert(A.shape == (4, 4))
        assert(B.shape == (4, 2))
        assert(F.shape == (2, 4))
        assert(h.size == num_agents * 4)
        assert(x0.size == num_agents * 4)

        self.CG = com_graph
        Q = nx.linalg.adjacency_matrix(com_graph).todense()
        D = np.diag([sum([1 for e in com_graph.edges() if e[0] == i]) for i in com_graph.nodes()])
        self.LG = np.dot(np.linalg.pinv(D), (D - Q))
        self.A = A
        self.B = B
        self.F = F
        self.x0 = x0
        self.h = h
        self.k = A[3,1]
        self.K = np.dot(self.B, self.F)
        self.T1 = np.kron(self.LG, self.K)

    def simulation_start(self):
        def dx_dt(t, x):
            Rz = self.compute_Rz(x, self.h)
            y = np.dot(np.kron(np.eye(self.N), self.A), x).reshape((len(x), 1)) + np.dot(self.T1, (x.reshape((len(x), 1)) - Rz))
            return np.asarray(y).reshape(len(x))

        self.ode = spi.ode(dx_dt)
        self.ode.set_integrator('vode', order=15, nsteps=5000, method='bdf')
        self.ode.set_initial_value(self.x0, 0)

    def simulation_step(self, dt):
        self.ode.integrate(self.ode.t + dt)
        return self.ode.y

    def compute_formation_quality(self, x):
        Rz = self.compute_Rz(x, self.h)
        h = np.array(Rz).reshape(len(Rz))

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

    def compute(self, T, dt):
        def dx_dt(t, x):
            Rz = self.compute_Rz(x, self.h)
            y = np.dot(np.kron(np.eye(self.N), self.A), x).reshape((len(x), 1)) + np.dot(self.T1, (x.reshape((len(x), 1)) - Rz))
            return np.asarray(y).reshape(len(x))

        ode = spi.ode(dx_dt)
        ode.set_integrator('vode', order=15, nsteps=5000, method='bdf')
        ode.set_initial_value(self.x0, 0)
        ys = []
        last_percent = 0
        while ode.successful() and ode.t < T:
            curr_percent = int((100 * ode.t) / T)
            if curr_percent > last_percent:
                print('%d%%' % curr_percent)
                last_percent = curr_percent
            ode.integrate(ode.t + dt)
            ys.append(ode.y)

        #ts = scipy.linspace(0, T, num=T/dt)
        #xys = scipy.integrate.odeint(dx_dt, self.x0, ts, full_output=0, mxstep=100)
        return np.vstack(ys)

    def compute_Rz(self, x, h):
        res = np.zeros((len(x), 1))
        h_column = h.reshape((len(x), 1))
        # for each agent...
        for i in range(self.N):
            Ei = np.zeros((self.N, self.N))
            Ei[i][i] = 1.0
            vi = np.array([x[4*i + 1], x[4*i + 3]])
            Rvi = self.rot_matrix(vi)
            Temp = np.kron(Ei, np.kron(Rvi, np.array([[1, 0], [0, 0]])))
            res += np.dot(Temp, h_column)
        return res

    def rot_matrix(self, v):
        '''Returns the two-dimensional rotation matrix 2x2 that
        rotates vector e1 to point in v direction'''
        #print(np.linalg.norm(v))
        t = np.array(v) / np.linalg.norm(v)
        return np.array([
            [t[0], -t[1]],
            [t[1], t[0]]
        ])

    def set_k(self, k):
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, -k],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, k,   0.0, 0.0]])
        self.k = k
        self.A = A

    @staticmethod
    def circular_from_com_graph(com_graph, h, x0, k, f1, f2):
        n = len(com_graph.nodes())
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, -k],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, k,   0.0, 0.0]])
        B = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 1.0]])
        F = np.array([[f1,  f2,  0.0, 0.0],
                      [0.0, 0.0, f1,  f2]])
        return OrientableModel(com_graph, n, A, B, F, h, x0)