import sys
from runner import *
from networks import *
from simulation import *
import numpy as np

def main():

    array_idx = 0
    if len(sys.argv) == 3:
        array_idx = int(sys.argv[2])
    
    assert len(sys.argv) > 1, "Must provide experiment file as argument"

    exp_filename = sys.argv[1]

    exp_params = load_exp_param_file(exp_filename)

    value, repeat = exp_values_from_index(exp_params, array_idx)

    print(f'(value, repeat) : ({value}, {repeat})')

    net = Network()
    
    # Set network variable
    setattr(net, exp_params['variable'], value)
    net.repeat = repeat

    sim_params = SimulationParameters()
    sim_params.sim_time_sec = 3
    sim_params.time_execution = True
    sim_params.inp_idxs = np.random.randint(0, net.N-1, net.N * 10 * sim_params.sim_time_sec)
    sim_params.inp_ts = np.random.randint(0, sim_params.sim_time_sec * MSPERSEC, net.N * 10 * sim_params.sim_time_sec)
    sim_params.voltages_to_save = np.array(net.N-1, dtype=np.int32)
    sim_params.delays_to_save = np.array([], dtype=np.int32)
    sim_params.variances_to_save = np.array([], dtype=np.int32)

    out = simulate(net, sim_params)

    print(out.sim_timer.get_percentages())

if __name__ == '__main__':
    main()
