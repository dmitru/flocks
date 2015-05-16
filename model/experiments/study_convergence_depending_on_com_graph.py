__author__ = 'dmitru'

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
from datetime import datetime

from Graphs import ComGraphUtils
from Utils import position_vector, FormationsUtil
from Models import LinearModel, OrientableModel
from Visualization import ModelAnimator

def run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol):
    model = OrientableModel.circular_from_com_graph(comGraph, h, x0, k, f1, f2)
    #a = ModelAnimator(model, draw_each_kth_frame=10, dt=0.01)
    #a.show()
    return measure_convergence(model, T, dt, tol)

def measure_convergence(model, T, dt=0.1, tol=1e-3):
    model.simulation_start()
    converged = False
    last_time_printed_step = None
    for step in range(int (T / dt)):
        y = model.simulation_step(dt)
        e = model.compute_formation_quality(y, dt)[0]
        print(step, e)
        if step > 1:
            e = e[0]
            now = datetime.now()
            if last_time_printed_step is None or (now - last_time_printed_step).seconds > 3:
                print(step, abs(e))
                last_time_printed_step = now
            if abs(e) < tol:
                print('Converged in %d steps (%f s.)' % (step, step * dt))
                converged = True
                return step
    if not converged:
        print('Model didn\'t converge in %d steps' % T / dt)
        return None

num_of_agents = 8
num_of_com_graphs = 100
num_of_experiments_per_graph = 20
v0 = (1, 0)
k = 0.7
f1 = -4.0
f2 = -4.0
dt = 0.001
tol = 0.005
T = 30

xs = []
ys = []
for i in range(num_of_com_graphs):
    num_edges = random.randint(num_of_agents,  ((num_of_agents-1)*num_of_agents)/2)
    print('Starting with com graph #%d/%d: (%d, %d)' % (i, num_of_com_graphs, num_of_agents, num_edges))
    comGraph = ComGraphUtils.random_graph(num_of_agents, num_edges)
    #for j in range(num_of_experiments_per_graph):
    h = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
    x0 = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
    convergence_steps = run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol)
    x = len(comGraph.edges())
    y = convergence_steps * dt if convergence_steps else None
    print(x, convergence_steps * dt if convergence_steps else '-')
    if y:
        xs.append(x)
        ys.append(y)

plt.scatter(xs, ys)
plt.show()

