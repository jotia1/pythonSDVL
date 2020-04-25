import sys
from networks import *
from simulation import *
from simtools import *
from visualisation import *
from metrics import *
import argparse


def run_experiment(sim_params_filename=DEFAULT_SIM_PARAMS_DICT_FILENAME, 
                    net_params_filename=DEFAULT_NET_PARAMS_DICT_FILENAME):

    net_params_dict = load_dict(net_params_filename)
    sim_params_dict = load_dict(sim_params_filename)
    sim_params_dict['net_params_dict'] = net_params_dict

    sim_params = SimulationParameters(sim_params_dict)
    net = Network(net_params_dict)

    sim_params.sim_time_sec = 11

    out = simulate(net, sim_params)

    #print(out.sim_timer.get_percentages())
    out.result = trueposxtrueneg(net, out, sim_params)
    save_experiment(net, out, sim_params)

    standard_plots(out)

    return net, out, sim_params


if __name__ == '__main__':
    run_experiment()
