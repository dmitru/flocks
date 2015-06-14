__author__ = 'dmitru'

import numpy as np

from Graphs import ComGraphUtils
from Utils import position_vector, FormationsUtil
from Models import LinearModel, OrientableModel
from Visualization import ModelAnimator

def run_experiment_with_params(comGraph, x0, h, vx0, vy0, k, f1, f2, T, p):
    model = OrientableModel.from_com_graph(comGraph, h, x0, k, f1, f2, D=1e4, orientable=True, acc=0, breaks=True, breaks_p=p)
    a = ModelAnimator(model, draw_each_kth_frame=10, dt=0.01, desired_pos=False)
    a.show()
    return measure_convergence(model, T)

def measure_convergence(model, T, tol=1e-3, dt=0.1):
    data = model.compute(T, dt, print_progress=True)
    converged = False
    for step in range(data.shape[0]):
        e = model.compute_formation_quality(data[step,:], dt)[0]
        if step > 1:
            e = e
            print(step, abs(e))
            if abs(e) < tol:
                print('Converged in %d steps (%f s.)' % (step, step * dt))
                converged = True
                return step
    if not converged:
        print('Model didn\'t converge in %d steps' % data.shape[0])
        return None

num_agents = 10
#comGraph = ComGraphUtils.random_graph(num_agents, num_agents, directed=False)
comGraph = ComGraphUtils.full_graph(num_agents)
h = np.kron(FormationsUtil.rotate_90c(FormationsUtil.random_positions(num_agents, 3.0)), np.array([1, 0]))
k = 0.0
f1 = -5.5
f2 = -5.5
p = 0.1

dt = 0.001

for vx0 in (0.0,):
    ones = np.ones(h.size / 4)
    vy0 = 6
    x0 = np.kron(FormationsUtil.random_positions(num_agents, 4.0), np.array([1, 0])) + \
        vx0 * np.kron(ones, np.array([0, 1, 0, 0])) + \
        vy0 * np.kron(ones, np.array([0, 0, 0, 1]))

    print(position_vector(x0), position_vector(h))

    convergence_steps = run_experiment_with_params(comGraph, x0, h, vx0, vy0, k, f1, f2, 30, p)
    print (vx0, convergence_steps * dt if convergence_steps else '-')

