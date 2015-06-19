import json
import traceback
from ParallelRunner import ParallelRunner

__author__ = 'dmitru'

import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
from networkx.readwrite import json_graph

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
    model = OrientableModel.from_com_graph(comGraph, h, x0, k, f1, f2, 0.0, breaks=breaks, breaks_p=breaks_p)
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

def gen_data(params):
    return gen_data_impl(*params)

def gen_data_impl(h, x0, num_of_agents, num_of_com_graphs):
    k = 0.7
    f1 = -4.0
    f2 = -4.0
    dt = 0.01
    tol = 0.005
    T = 30

    xs = []
    ys = []
    for i in range(generate_data.py):
        num_edges = random.randint(num_of_agents, ((num_of_agents - 1)*num_of_agents)/2)
        print('Starting with com graph #%d/%d: (%d, %d)' % (i, num_of_com_graphs, num_of_agents, num_edges))
        comGraph = ComGraphUtils.random_graph(num_of_agents, num_edges, directed=False)
        ys_experiments = []
        convergence_steps = run_experiment_with_params(comGraph, x0, h, k, f1, f2, T, dt, tol, False, 0)
        y = convergence_steps * dt if convergence_steps else None
        print('%d/%d' % (i, num_of_com_graphs))
        print(num_edges, convergence_steps * dt if convergence_steps else '-')
        if y:
            ys_experiments.append(y)

        if len(ys_experiments) > 0:
            xs.append(json_graph.node_link_data(comGraph))
            ys.append(sum(ys_experiments) / len(ys_experiments))

    return xs, ys

def gen_data_test():
    num_of_agents = 8
    num_of_com_graphs = 2
    v0 = (10, 10)
    h = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
    x0 = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
    result = gen_data((h, x0, num_of_agents, num_of_com_graphs))
    pass

def run_gen_data():
    parallel = True

    if parallel:
        cluster = ParallelRunner('flocks')
        cluster.execute(['from functools import wraps',
                        'import errno',
                        'import os',
                        'import signal',
                        'import networkx as nx',
                        'from networkx.readwrite import json_graph'])

        cluster.push(dict(gen_data=gen_data,
                          gen_data_impl=gen_data_impl,
                          measure_convergence=measure_convergence,
                          run_experiment_with_params=run_experiment_with_params,
                          TimeoutError=TimeoutError,
                          timeout=timeout))

    num_experiments = 10

    for exper_iter in range(num_experiments):
        print 'EXPERIMENT %d/%d' % (exper_iter, num_experiments)
        now = datetime.now()
        num_of_agents = random.randint(10, 20)
        num_of_com_graphs = 10
        num_edges = random.randint(num_of_agents, num_of_agents * (num_of_agents - 1))
        experiment_name = 'fixed_params_%d_agents_%d_edges_' % (num_of_agents, num_edges)
        suffix = '%d_%d_%d__%d_%d_%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        K = 10
        #num_of_agents = random.randint(10, 20)
        #num_of_com_graphs = 10
        #num_edges = random.randint(num_of_agents, num_of_agents * (num_of_agents - 1))
        v0 = (10, 10)
        h = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])
        x0 = FormationsUtil.extend_position_to_vpv(FormationsUtil.random_positions(num_of_agents), v0[0], v0[1])

        if parallel:
            cluster.dview['h'] = h
            cluster.dview['x0'] = x0
        try:
            for k in range(K):
                ExperimentManager.init('data_%d' % k, root_path='/home/dmitry/Music/flocks/data/fixed_params/%s%s_%d' % (experiment_name, suffix, exper_iter))
                print 'FINISHED: %d/%d' % (k, K)
                com_graph = ComGraphUtils.random_graph(num_of_agents, num_edges, directed=True)

                if parallel:
                    cluster.dview['com_graph'] = com_graph
                args = ((h, x0, num_of_agents, num_of_com_graphs) for i in range(8))

                if parallel:
                    results = cluster.lbview.map(gen_data, args)
                    results = list(map(lambda x: (None, None) if x is None else x, results))
                else:
                    results = map(gen_data, args)
                xs = []
                ys = []
                for result in results:
                    xs += [result[0]]
                    ys += [result[1]]
                xs = sum(xs, [])
                ys = sum(ys, [])
                f_params = open(ExperimentManager.next_filename(suffix='params', extension='.txt', increment=False), 'w')
                f_params.write(json.dumps('experiment: %d\nnum_of_agents: %d\nnum_of_edges: %d' % (exper_iter, num_of_agents, num_edges)))
                f_params.close()
                results = []
                for i in range(len(xs)):
                    results.append(
                        {
                            'graph': xs[i],
                            'results': [ys[i]]
                         }
                    )
                f_graph = open(ExperimentManager.next_filename(suffix='results', extension='.txt'), 'w')
                f_graph.write(json.dumps(results))
                f_graph.close()
        except Exception, e:
            print 'Error in experiment %d, k = %d: %s' % (exper_iter, k, traceback.format_exc())

if __name__ == '__main__':
    run_gen_data()



