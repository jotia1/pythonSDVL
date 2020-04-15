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
    array_idx = 1
    exp_filename = 'LOCAL'
    exp_params = {
        'variable'      :   'fgi',
        'var_min'       :   0.0226,
        'var_max'       :   0.0227,
        'var_step'      :   0.0001,
        'repeats'       :   3,
        'job_name'      :   'LOCALJOB',
        'output_folder' :   'LOCAL',
    }

    assert len(sys.argv) < 4, "Too Many input arguments."

    if len(sys.argv) > 1:
        exp_filename = sys.argv[1]
        exp_base_filename = sys.argv[1].strip('.json')
        exp_params = load_exp_param_file(exp_filename)

    if len(sys.argv) > 2:
        array_idx = int(sys.argv[2])
        # Update output filename
        exp_params['output_folder'] = f"{exp_base_filename}"
    
    value, repeat = exp_values_from_index(exp_params, array_idx)

    net = Network()
    
    # Set network variable
    setattr(net, exp_params['variable'], value)
    net.repeat = repeat

    sim_params = SimulationParameters()
    sim_params.array_idx = array_idx
    _,_,slurmid = exp_params['output_folder'].split('_')
    sim_params.output_folder = f'{exp_params["job_name"]}_{slurmid}'
    sim_params.exp_base_filename = f'{sim_params.output_folder}_{array_idx}'
    sim_params.sim_time_sec = 10
    sim_params.time_execution = False #True
    #sim_params.inp_idxs, sim_params.inp_ts = get_input(net, sim_params)
    sim_params.p_inp = np.arange(0, 500)
    sim_params.p_ts = np.reshape(np.tile(np.arange(0, 50), (10, 1)), (-1), order='F')
    sim_params.data_fcn = lambda : embedded_pattern(50, 10, 2000, 500, 5, sim_params.p_inp, sim_params.p_ts, None, 0.0)
    sim_params.voltages_to_save = np.array([net.N-1], dtype=np.int32)
    sim_params.delays_to_save = np.array([], dtype=np.int32)
    sim_params.variances_to_save = np.array([], dtype=np.int32)

    #print(sim_params.inp_idxs.shape, sim_params.inp_ts.shape)
    out = simulate(net, sim_params)

    # To save space, remove data spikes
    out.spike_time_trace = out.spike_time_trace[out.spike_time_trace[:, 1] == net.N-1, :]

    #print(out.sim_timer.get_percentages())
    out.result = trueposxtrueneg(net, out, sim_params)
    save_experiment(net, out, sim_params)

    #nout = load_experiment(f'{sim_params.output_folder}/{sim_params.exp_base_filename}.npz')

    #standard_plots(nout)

def save_experiment(net, out, sim_params):
    # spike_time_trace, vt, iapp, offsets, 
    np.savez(f'{sim_params.output_folder}/{sim_params.exp_base_filename}.npz', 
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
