__author__ = 'dmitru'

import numpy as np

from Graphs import ComGraphUtils
from Utils import position_vector, FormationsUtil
from Models import LinearModel, OrientableModel
from Visualization import ModelAnimator

n = 4

# 1. com graph
comGraph = ComGraphUtils.full_graph(n)

# 2. choose desired formation
h = np.kron(FormationsUtil.rotate_90c(FormationsUtil.square_4()), np.array([1, 0]))

# 3. set up initial state
ones = np.ones(h.size / 4)
vx0 = -1.0
vy0 = -0.0
x0 = np.kron(np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]), np.array([1, 0])) + \
    vx0 * np.kron(ones, np.array([0, 1, 0, 0])) + \
    vy0 * np.kron(ones, np.array([0, 0, 0, 1]))

# 4. choose other model parameters
k = 0.0
f1 = -4.0
f2 = -4.0

model = OrientableModel.circular_from_com_graph(comGraph, h, x0, k, f1, f2)

# tol = 1e-3
# dt = 0.1
# data = model.compute(10, dt)
# converged = False
# e = None
# for step in range(data.shape[0]):
#     e = FormationsUtil.compute_closeness(h, data[step,:])
#     print(step, abs(e))
#     if abs(e) < tol:
#         print('Converged in %d steps (%f s.)' % (step, step * dt))
#         converged = True
#         break
# if not converged:
#     print('Model didn\'t converge in %d steps' % data.shape[0])

a = ModelAnimator(model, draw_each_kth_frame=10, dt=0.01)
a.show()