import sys
from runcluster import *
from networks import *
from simulation import *
from datasource import *
from visualisation import *
import numpy as np
from metrics import *


def main():
    run_simulation()

def run_simulation():

    exp_params, task_id, slurm_id = get_exp_params()

    value, repeat = exp_values_from_index(exp_params, task_id)

    net = Network()
    
    # Set network variable
    setattr(net, exp_params['variable'], value)
    net.repeat = repeat

    sim_params = SimulationParameters(exp_params, slurm_id, task_id)

    #print(sim_params.inp_idxs.shape, sim_params.inp_ts.shape)
    out = simulate(net, sim_params)

    # To save space, remove data spikes
    out.spike_time_trace = out.spike_time_trace[out.spike_time_trace[:, 1] == net.N-1, :]

    #print(out.sim_timer.get_percentages())
    out.result = trueposxtrueneg(net, out, sim_params)
    save_experiment(net, out, sim_params)

    #nout = load_experiment(f'{sim_params.output_folder}/{sim_params.exp_base_filename}.npz')

    #standard_plots(nout)

def get_exp_params():
    """ Return the exp_params dict object and the task ID
    """
    array_idx = 1
    slurm_id = 1
    exp_params = {
        'variable'      :   'fgi',
        'var_min'       :   0.0226,
        'var_max'       :   0.0227,
        'var_step'      :   0.0001,
        'repeats'       :   3,
        'job_name'      :   'LOCALJOB',
        'running_ntasks':   1,
    }

    assert len(sys.argv) < 4, "Too Many input arguments."

    if len(sys.argv) > 1:
        exp_filename = sys.argv[1]
        exp_params = load_exp_param_file(exp_filename)
        _,_,slurm_id = exp_filename.strip('.json').split('_')

    if len(sys.argv) > 2:
        array_idx = int(sys.argv[2])
    
    return exp_params, array_idx, slurm_id

def save_experiment(net, out, sim_params):
    # spike_time_trace, vt, iapp, offsets, 
    np.savez(f'{sim_params.output_folder}/{sim_params.output_base_filename}.npz', 
                                    spike_time_trace=out.spike_time_trace,
                                    vt=out.vt,
                                    result=out.result,
                                    iapp_trace=out.iapp_trace,
                                    offsets=out.offsets)

def load_experiment(filename):
    npz = np.load(filename)
    out = SimulationOuput()
    out.spike_time_trace = npz['spike_time_trace']
    out.vt = npz['vt']
    out.result = npz['result']
    out.iapp_trace = npz['iapp_trace']
    out.offsets = npz['offsets']
    return out


def get_input(net, sim_params):
    inp = np.array([])
    ts = np.array([])
    p_inp = np.arange(0, 500)
    p_ts = np.reshape(np.tile(np.arange(0, 50), (10, 1)), (-1), order='F')
    for sec in range(sim_params.sim_time_sec):
        sec_inp, sec_ts = embedded_pattern(50, 10, 2000, 500, 5, p_inp, p_ts, None, 0.0)
        inp = np.concatenate((inp, sec_inp))
        ts = np.concatenate((ts, sec_ts + (sec * MSPERSEC)))

    return inp.astype(int), ts.astype(int)

if __name__ == '__main__':
    main()
