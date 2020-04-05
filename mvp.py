import numpy as np
from visualisation import * 
from simulation import *
from networks import *

def main():
    net = Network()
    sim_params = SimulationParameters()
    out, sim_timer = simulate(net, sim_params)

    print(sim_timer.get_percentages())
    standard_plots(out.spike_time_trace, out.iapp_trace, out.vt)


if __name__ == '__main__':
    main()