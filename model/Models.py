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
    def __init__(self, com_graph, num_agents, A, B, F, h, x0, D=1e4, r0=0.5, r1=0.01):
        '''Parameters:
        com_graph - communication graph that agents use,
        agents - list of Agent objects,
        A - 4x4 matrix, B - 4x2 matrix, see [1].II for details
        F - 2x4 feedback matrix
        h - desired formation
        x0 - initial position-velocity vector of all agents
        D=1e4, r0=1.0, r1=0.01 - parameters of repulsion
        (see "Outdoor Flocking ...", pp. 5-6)
        '''
        self.N = num_agents
        assert(type(com_graph) == nx.DiGraph)
        assert(len(com_graph.nodes()) == self.N)

        assert(A.shape == (4, 4))
        assert(B.shape == (4, 2))
        assert(F.shape == (2, 4))
        assert(h.size == num_agents * 4)
        assert(x0.size == num_agents * 4)

        self.text_to_show = None
        self.CG = com_graph
        Q = nx.linalg.adjacency_matrix(com_graph).todense()
        Diag = np.diag([sum([1 for e in com_graph.edges() if e[0] == i]) for i in com_graph.nodes()])
        self.LG = np.dot(np.linalg.pinv(Diag), (Diag - Q))
        self.A = A
        self.B = B
        self.F = F
        self.x0 = x0
        self.h = h
        self.k = A[3,1]
        self.K = np.dot(self.B, self.F)
        self.T1 = np.kron(self.LG, self.K)
        self.delta_prev = None
        self.D = D
        self.r0 = r0
        self.r1 = r1

    def simulation_start(self):
        def dx_dt(t, x):
            Rz = self.compute_Rz(x, self.h)

            # basic dynamics of the model
            y = np.dot(np.kron(np.eye(self.N), self.A), x).reshape((len(x), 1)) + np.dot(self.T1, (x.reshape((len(x), 1)) - Rz))
            y = np.asarray(y).reshape(len(x))

            # TODO: make it a model's parameters
            D = self.D
            r0 = self.r0
            r1 = self.r1

            y_rep = np.zeros(y.shape)
            # calculate repulsive acceleration
            for i in range(self.N):
                a_i = np.array([0.0, 0.0])
                for j in range(self.N):
                    if i == j:
                        continue
                    x_i = np.array([x[i * 4], x[i * 4 + 2]])
                    x_j = np.array([x[j * 4], x[j * 4 + 2]])
                    x_ij = x_j - x_i
                    x_ij_norm = np.linalg.norm(x_ij)
                    if x_ij_norm < r0:
                        a_i = a_i -D * min(r1, r0 - x_ij_norm) * x_ij / x_ij_norm

                y_rep[i * 4 + 1] = a_i[0]
                y_rep[i * 4 + 3] = a_i[1]

            print(np.linalg.norm(y), np.linalg.norm(y_rep))
            y += y_rep
            return y

        self.ode = spi.ode(dx_dt)
        self.ode.set_integrator('vode', order=15, nsteps=5000, method='bdf')
        self.ode.set_initial_value(self.x0, 0)

    def simulation_step(self, dt):
        self.ode.integrate(self.ode.t + dt)
        return self.ode.y

    def compute_formation_quality(self, x, dt):
        ones = np.ones(self.h.size / 4)
        es = [np.kron(ones, np.array([1,0,0,0])),
              np.kron(ones, np.array([0,1,0,0])),
              np.kron(ones, np.array([0,0,1,0])),
              np.kron(ones, np.array([0,0,0,1]))]
        x_relative = np.array(x)
        for i in range(4):
            x_relative -= x[i] * es[i]

        rot = np.matrix(self.rot_matrix((x[1], x[3])))

        temp = []
        for i in range(self.h.size // 4):
            x_agent = (x_relative[4*i], x_relative[4*i + 2])
            t = np.dot(x_agent, rot)
            temp.append(t[0,0])
            temp.append(t[0,1])
        temp = np.array(temp)
        delta = temp

        #return res

        #result1 = None if self.delta_prev is None else np.linalg.norm(position_vector(self.delta_prev - delta) / dt)
        #result2 = None if self.delta_prev is None else np.linalg.norm(position_vector(self.compute_Rz(x, (self.delta_prev - delta) / dt)))
        #result3 = None if self.delta_prev is None else np.linalg.norm(velocity_vector(self.delta_prev - delta) / dt)
        #result4 = None if self.delta_prev is None else np.linalg.norm(velocity_vector(self.compute_Rz(x, (self.delta_prev - delta) / dt)))

        result = (np.linalg.norm(self.delta_prev - delta), ) if self.delta_prev is not None else None
        self.text_to_show = '%.6f %s' % (dt, str(self.delta_prev - delta) if self.delta_prev is not None else '')

        self.delta_prev = delta

        return (result, ) if result is not None else (0, )

        return (np.linalg.norm(delta), result1, result2, result3, result4) #np.linalg.norm(delta), (None if result is None else np.linalg.norm(result)), \


        #return np.linalg.norm(delta)

    def compute(self, T, dt, print_progress=False):
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
            if print_progress and curr_percent > last_percent:
                print('%d%%' % curr_percent)
                last_percent = curr_percent
            ode.integrate(ode.t + dt)
            ys.append(ode.y)

        #ts = scipy.linspace(0, T, num=T/dt)
        #xys = scipy.integrate.odeint(dx_dt, self.x0, ts, full_output=0, mxstep=100)
        return np.vstack(ys)

    def compute_Rz(self, x, h):
        assert x.size == h.size
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