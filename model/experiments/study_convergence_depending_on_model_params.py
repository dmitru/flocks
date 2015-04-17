__author__ = 'dmitru'

import numpy as np

from Graphs import ComGraphUtils
from Utils import position_vector, FormationsUtil
from Models import LinearModel, OrientableModel
from Visualization import ModelAnimator

def run_experiment_with_params(comGraph, x0, h, vx0, vy0, k, f1, f2, T):
    model = OrientableModel.circular_from_com_graph(comGraph, h, x0, k, f1, f2)
    a = ModelAnimator(model, draw_each_kth_frame=10, dt=0.01)
    a.show()
    return measure_convergence(model, T)

def measure_convergence(model, T, tol=1e-3, dt=0.1):
    data = model.compute(T, dt, print_progress=False)
    converged = False
    for step in range(data.shape[0]):
        e = model.compute_formation_quality(data[step,:], dt)[0]
        if step > 1:
            print(step, abs(e))
            if abs(e) < tol:
                print('Converged in %d steps (%f s.)' % (step, step * dt))
                converged = True
                return step
    if not converged:
        print('Model didn\'t converge in %d steps' % data.shape[0])
        return None

comGraph = ComGraphUtils.full_graph(6)
h = np.kron(FormationsUtil.rotate_90c(FormationsUtil.arrow_tip_6()), np.array([1, 0]))
k = 0.00
f1 = -4.0
f2 = -4.0

dt = 0.001

for vx0 in (3.0,):
    ones = np.ones(h.size / 4)
    vy0 = 0
    x0 = np.kron(np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0]), np.array([1, 0])) + \
        vx0 * np.kron(ones, np.array([0, 1, 0, 0])) + \
        vy0 * np.kron(ones, np.array([0, 0, 0, 1]))

    convergence_steps = run_experiment_with_params(comGraph, x0, h, vx0, vy0, k, f1, f2, 30)
    print (vx0, convergence_steps * dt if convergence_steps else '-')

