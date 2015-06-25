import os
from networkx.readwrite import json_graph
import numpy as np

# Import standard Python libraries
import json
import traceback
import random
from datetime import datetime

# Import project libraries
from Models import OrientableModel
from ParallelRunner import ParallelRunner
from Graphs import ComGraphUtils
from Utils import FormationsUtil, ExperimentManager

##
# Experiment description
# One run consists of a number of experiments
# Each experiment sweeps one model parameter while holding others fixed
# Results are saved like this:
# RESULT_ROOT/RUN_NAME_*/experiment_*/
# results are saved as JSON files in the following format:
# {
#    model_params: <all model params>,
#    results:      <array of results, currently it contains only one number: number of
#                   steps until convergence>,
#    experiment_tag: <integer, from 1 to NUM_OF_EXPERIMENTS>
# }
##

RESULT_ROOT        = '/home/dmitry/flocks_results/'
RUN_NAME           = 'graph_optimization_n_e_grid_algo'

ns = list(range(6, 20, 2))
n_es = []
for n in ns:
    es_n = range(n, int(n * (n-1) / 2), 2)
    n_es += [(n, es) for es in es_n]

NUM_OF_EXPERIMENTS = len(n_es)
parallel           = True
save_results       = True

sweep_param_name = 'com_graph'
#sweep_param_options = [(v0, 0.0) for v0 in np.arange(1.5, 10, 0.1)] # v0
#sweep_param_options = np.arange(0.0, 14.0, 0.2)
sweep_param_options = []

def prepare_experiment_params(fixed_params, sweep_params):
    # Set up params that are fixed during the experiment
    params = dict(fixed_params)
    for key in sweep_params:
        params[key] = sweep_params[key]
    return params

# Global variables, do not change these lines
fixed_params = None
params = None

#num_of_agents = random.randint(14, 24)
#num_edges = random.randint(num_of_agents * 3, num_of_agents * (num_of_agents - 1) / 2)

def run_gen_data():
    global fixed_params, sweep_param_options, n_es

    if parallel:
        cluster = init_cluster()

    now = datetime.now()
    run_dirsuffix = '%d_%d_%d__%d_%d_%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    run_dirname = os.path.join(RESULT_ROOT, '%s_%s' % (RUN_NAME, run_dirsuffix))

    for exper_iter in range(1, NUM_OF_EXPERIMENTS + 1):
        num_edges = n_es[exper_iter][1]
        num_of_agents = n_es[exper_iter][0]
        print 'EXPERIMENT %d/%d' % (exper_iter, NUM_OF_EXPERIMENTS)
        now = datetime.now()
        experiment_dirsuffix = '%d_%d_%d__%d_%d_%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        experiment_dirname = 'experiment_%d_%s_%s__%s_%s' % (exper_iter, sweep_param_name, experiment_dirsuffix,
                                                             num_of_agents, num_edges)

        sweep_param_options = []
        for i in range(1000):
            com_graph = ComGraphUtils.best_ac_graph(num_of_agents, num_edges)
            sweep_param_options.append(com_graph)
        # Params that are held fixed during the experiment
        fixed_params = dict(
            v0 = (10.0, 10.0),
            k = 0,
            x0 = FormationsUtil.random_positions(num_of_agents, seed=123),
            h = FormationsUtil.random_positions(num_of_agents, seed=452),
            f1 = -4.0,
            f2 = -4.0,
            dt = 0.01,
            tol = 0.002,
            breaks = False,
            breaks_p = 0.0,
            T = 30
        )

        args = []
        for sweep_param in sweep_param_options:
            args.append(sweep_param)

        if parallel:
            cluster.dview['fixed_params'] = fixed_params
        try:
            if save_results:
                ExperimentManager.init(experiment_dirname, root_path=run_dirname)

            # Prepare arguments for execution
            args = []
            for sweep_param in sweep_param_options:
                args.append({sweep_param_name: sweep_param})

            if parallel:
                print 'Launching %d jobs in parallel...' % len(args)
                args_chunks = []
                cur_chunk = []
                i = 0
                while i < len(args):
                    cur_chunk.append(args[i])
                    if len(cur_chunk) >= 21:
                        args_chunks.append(cur_chunk)
                        cur_chunk = []
                    i += 1
                results = []
                for i, args_chunk in enumerate(args_chunks):
                    results_chunk = cluster.lbview.map(gen_data, args_chunk)
                    print 'Finished %d/%d job chunks' % (i, len(args_chunks))
                    results += results_chunk
            else:
                results = map(gen_data, args)
            results = filter(lambda res: res['results'] is not None, results)

            # Save the results
            if save_results:
                fout = open(ExperimentManager.next_filename(suffix='results', extension='.txt', increment=False), 'w')
                transformed_results = serialize_results(results, exper_iter)
                fout.write(json.dumps(transformed_results))
                fout.close()
        except Exception, e:
            print 'Error in experiment %d: %s' % (exper_iter, traceback.format_exc())

def gen_data(sweep_params):
    complete_params = prepare_experiment_params(fixed_params, sweep_params)
    convergence_steps = run_experiment_with_params(**complete_params)
    y = convergence_steps * complete_params['dt'] if convergence_steps else None
    return dict(model_params=complete_params, results=[y] if y else None)

def init_cluster():
    cluster = ParallelRunner('flocks')
    cluster.execute(['from functools import wraps',
                     'import errno',
                     'import os',
                     'import signal',
                     'import networkx as nx',
                     'from networkx.readwrite import json_graph'])
    cluster.push(dict(gen_data=gen_data,
                      measure_convergence=measure_convergence,
                      run_experiment_with_params=run_experiment_with_params,
                      prepare_experiment_params=prepare_experiment_params))
    return cluster

def run_experiment_with_params(com_graph, v0, x0, h, k, f1, f2, T, dt, tol, breaks, breaks_p, **kwargs):
    x0 = FormationsUtil.extend_position_to_vpv(x0, v0[0], v0[1])
    h = FormationsUtil.extend_position_to_vpv(h, 0, 0)
    model = OrientableModel.from_com_graph(com_graph, h, x0, k, f1, f2, 0.0, breaks=breaks, breaks_p=breaks_p)
    t1 = datetime.now()
    res = measure_convergence(model, T, dt, tol)
    t2 = datetime.now()
    print 'Simulation time: %d ms' % (int((t2 - t1).microseconds / 1e3))
    return res


def measure_convergence(model, T, dt=0.1, tol=1e-3):
    model.simulation_start()
    converged = False
    last_time_printed_step = None
    for step in range(int(T / dt)):
        y, rel_h = model.simulation_step(dt)
        e = model.compute_formation_quality(y, dt)[0]
        # print(step, e)
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


def serialize_results(results, experiment_tag):
    new_results = []
    for result in results:
        new_result = dict(result)
        new_result['model_params']['com_graph'] = json_graph.node_link_data(result['model_params']['com_graph'])
        new_result['experiment_tag'] = experiment_tag
        new_results.append(new_result)
    return new_results


if __name__ == '__main__':
    run_gen_data()
