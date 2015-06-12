import traceback
from ParallelRunner import ParallelRunner

__author__ = 'dmitru'

import matplotlib
matplotlib.use('TkAgg')

from pylab import rcParams
rcParams['figure.figsize'] = 18.5, 10.5

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

from Graphs import ComGraphUtils
from Utils import position_vector, FormationsUtil, laplace_matrix, ExperimentManager, get_cmap
from Models import LinearModel, OrientableModel
from Visualization import ModelAnimator

x0, h, com_graph = None, None, None

from functools import wraps
import errno
import os
import signal

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

def run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol, breaks, breaks_p):
    model = OrientableModel.circular_from_com_graph(comGraph, h, x0, k, f1, f2, 0.0, breaks=breaks, breaks_p=breaks_p)
    #a = ModelAnimator(model, draw_each_kth_frame=10, dt=0.01)
    #a.show()
    t1 = datetime.now()
    res = measure_convergence(model, T, dt, tol)
    t2 = datetime.now()
    print 'Simulation time: %d ms' % (int((t2 - t1).microseconds / 1e3))
    return res

def measure_convergence(model, T, dt=0.1, tol=1e-3):
    model.simulation_start()
    converged = False
    last_time_printed_step = None
    for step in range(int (T / dt)):
        y ,rel_h = model.simulation_step(dt)
        e = model.compute_formation_quality(y, dt)[0]
        #print(step, e)
        if step > 1:
            e = e
            now = datetime.now()
            if last_time_printed_step is None or (now - last_time_printed_step).seconds > 3:
                if last_time_printed_step is not None:
                    print(step, abs(e))
                last_time_printed_step = now
            if abs(e) < tol:
                print('Converged in %d steps (%f s.)' % (step, step * dt))
                converged = True
                return step
    if not converged:
        print('Model didn\'t converge in %d steps' % (T / dt))
        return None

def measure_speed_vs_number_of_edges():
    num_of_agents = 20
    num_of_com_graphs = 14
    num_of_experiments_per_graph = 4
    v0 = (10, 10)
    k = 0.7
    f1 = -4.0
    f2 = -4.0
    dt = 0.01
    tol = 0.0005
    T = 30

    xs = []
    ys = []
    for i in range(num_of_com_graphs):
        num_edges = random.randint(num_of_agents, ((num_of_agents - 1)*num_of_agents)/2)
        print('Starting with com graph #%d/%d: (%d, %d)' % (i, num_of_com_graphs, num_of_agents, num_edges))
        comGraph = ComGraphUtils.random_graph(num_of_agents, num_edges, directed=False)
        ys_experiments = []
        for j in range(num_of_experiments_per_graph):
            h = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
            x0 = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
            convergence_steps = run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol)
            y = convergence_steps * dt if convergence_steps else None
            print('%d/%d  %d/%d' % (i, num_of_com_graphs, j, num_of_experiments_per_graph))
            print(num_edges, convergence_steps * dt if convergence_steps else '-')
            if y:
                ys_experiments.append(y)
        if len(ys_experiments) > 0:
            xs.append(num_edges)
            ys.append(sum(ys_experiments) / len(ys_experiments))

    return xs, ys

def measure_speed_vs_break_p_fixed_graph(p):
    global x0, h, com_graph
    k = 0.2
    f1 = -4.0
    f2 = -4.0
    dt = 0.01
    tol = 0.01
    T = 80
    try:
        res = run_experiment_with_params(com_graph, x0, h, k, f1, f2, T, dt, tol, True, p)
    except TimeoutError, e:
        print 'run_experiment_with_params timed out'
        return None
    return (p, res)

def measure_speed_vs_break_p_fixed_graph_par(args):
    return measure_speed_vs_break_p_fixed_graph(*args)

def measure_speed_vs_first_eigenvalue_undirected_par(args):
    return measure_speed_vs_first_eigenvalue_undirected(*args)

def measure_speed_vs_first_eigenvalue_undirected(num_of_agents, num_of_com_graphs, x0, h):
    k = 0.0
    f1 = -4.0
    f2 = -4.0
    dt = 0.01
    tol = 0.01
    T = 30

    xs = []
    ys = []
    for i in range(num_of_com_graphs):
        num_edges = random.randint(num_of_agents, ((num_of_agents - 1)*num_of_agents)/2)
        print('Starting with com graph #%d/%d: (%d, %d)' % (i, num_of_com_graphs, num_of_agents, num_edges))
        comGraph = ComGraphUtils.random_graph(num_of_agents, num_edges, directed=False)
        convergence_steps = run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol)
        L = laplace_matrix(comGraph)
        assert np.max(L - L.transpose()) < 1e-6
        eigenvalues = sorted(np.linalg.eigvals(L))
        x = np.real(eigenvalues[1]).tolist()

        assert np.imag(eigenvalues[1]) < 1e-6
        y = convergence_steps * dt if convergence_steps else None
        print('%d/%d' % (i, num_of_com_graphs))
        print(x, convergence_steps * dt if convergence_steps else '-')
        if y:
            xs.append(x)
            ys.append(y)
    return xs, ys

if __name__ == '__main__':
    ExperimentManager.init('breaking_links', root_path='/home/dmitry/Music/flocks/results')

    parallel = True

    if parallel:
        cluster = ParallelRunner('flocks')
        cluster.execute(['from functools import wraps',
                        'import errno',
                        'import os',
                         'import signal'])

        cluster.push(dict(measure_speed_vs_first_eigenvalue_undirected=measure_speed_vs_first_eigenvalue_undirected,
                          measure_speed_vs_break_p_fixed_graph=measure_speed_vs_break_p_fixed_graph,
                          measure_convergence=measure_convergence,
                          run_experiment_with_params=run_experiment_with_params,
                          TimeoutError=TimeoutError,
                          timeout=timeout))

    num_experiments = 1

    for exper_iter in range(num_experiments):
        print 'EXPERIMENT %d/%d' % (exper_iter, num_experiments)
        K = 10
        cmap = get_cmap(K + 1)
        num_of_agents = random.randint(8, 14)
        num_edges = random.randint(num_of_agents, num_of_agents * (num_of_agents - 1))
        v0 = (10, 10)
        h = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
        x0 = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])

        if parallel:
            cluster.dview['h'] = h
            cluster.dview['x0'] = x0
        try:
            XS = []
            YS = []
            for k in range(K):
                print 'FINISHED: %d/%d' % (k, K)
                com_graph = ComGraphUtils.random_graph(num_of_agents, num_edges, directed=True)

                if parallel:
                    cluster.dview['com_graph'] = com_graph
                ps = np.arange(0.4, 0.90, 0.01)
                args = [(p, ) for p in ps]

                if parallel:
                    results = cluster.lbview.map(measure_speed_vs_break_p_fixed_graph_par, args)
                    results = list(map(lambda x: (None, None) if x is None else x, results))
                else:
                    results = map(measure_speed_vs_break_p_fixed_graph_par, args)
                xs = []
                ys = []
                for result in results:
                    print result
                    xs += [result[0]]
                    ys += [result[1]]
                plt.plot(xs, ys, color=cmap(k))
                XS.append(xs)
                YS.append(ys)
            plt.savefig(ExperimentManager.next_filename(increment=False), dpi=200)
            plt.clf()
            f_params = open(ExperimentManager.next_filename(extension='.txt', increment=False), 'w')
            f_params.write('experiment: %d\nnum_of_agents: %d\nnum_of_edges: %d' % (exper_iter, num_of_agents, num_edges))
            f_params.close()
            np.savetxt(ExperimentManager.next_filename(extension='.csv'), np.vstack((np.array(XS), np.array(YS))), delimiter=",")
        except Exception, e:
            print 'Error in experiment %d, k = %d: %s' % (exper_iter, k, traceback.format_exc())


