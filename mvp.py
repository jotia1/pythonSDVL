import numpy as np
from visualisation import * 
from simulation import *
from networks import *

def main():
    net = Network()
    sim_params = SimulationParameters()
    #out, sim_timer = simulate(net, sim_params)

    #print(sim_timer.get_percentages())
    #standard_plots(out.spike_time_trace, out.iapp_trace, out.vt)



def plot_simple_sdvl_example():
    net = Network(n_inputs=3)
    net.fgi = 8
    net.a1 = 1
    net.a2 = 1
    net.b1 = 3
    net.b2 = 3

    sim_params = SimulationParameters()
    sim_params.sim_time_sec = 30
    sim_params.time_execution = True
    presentation_freq = 2
    num_presentations = sim_params.sim_time_sec * presentation_freq
    sim_params.inp_idxs = np.tile([0, 1, 2], num_presentations)
    sim_params.inp_ts = np.tile(list(range(num_presentations)), (3, 1)).reshape(-1, order='F') * 500\
                        + np.tile([0, 3, 7], num_presentations)
    sim_params.voltages_to_save = np.array([3])
    sim_params.delays_to_save = np.array([3])
    sim_params.variances_to_save = np.array([3])
                      
    out = simulate(net, sim_params)

    print(out.sim_timer.get_percentages())

    variance_precision = 0.01
    current_steps = 40
    var_range = np.concatenate((np.arange(net.variance_min, net.variance_max, variance_precision), [10]))
    ptable = getlookuptable(var_range,
                            np.arange(0, net.delay_max),
                            np.arange(0, current_steps),
                            net.fgi)
    
    sample_times = [1, 2000, 5000, 30000]
    for axi, sample_time in enumerate(sample_times):
        indexs_delays = np.round(out.delayst[0:3, 0, sample_time-1]) - PTABLEDELAYINDEXOFFSET
        indexs_var = np.round(out.variancest[0:3, 0, sample_time-1] / variance_precision - PTABLEVARIANCEINDEXOFFSET)

        p_values = ptable[indexs_var.astype(int), indexs_delays.astype(int), :]

        ax = plt.subplot(2, 2, axi+1)
        total_current = np.zeros(40)
        times = [0, 3, 7]
        for i in range(3):
            sample = np.concatenate((np.zeros(times[i]), p_values[i, 0:-times[i] or None]))
            ax.plot(sample[:21] + 0.2, label=f'Input #{i+1}')
            total_current += sample
        ax.plot(total_current[:21] + 0.2, label=f'Summed current')
        ax.set_ylim((0, 17))
        ax.set_facecolor((0,0,0,0))
        if axi in [0, 1]:
            ax.xaxis.set_visible(False)
        if axi in [1, 3]:
            ax.yaxis.set_visible(False)
        if axi in [0, 2]:
            ax.set_ylabel('Current')
            ax.set_yticklabels([str(x) for x in range(0, 20, 5)])
        if axi in [2, 3]:
            #ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.set_xlabel('Time (ms)')
        if axi == 1:
            ax.legend()
        ax.set_title(f'After {sample_time // 1000} seconds of training')
    
    plt.show()

    ## 
    #out_firing_times = out.spike_time_trace[out.spike_time_trace[:, 1] == 3, 0]
    #out_offsets = out_firing_times % 500
    #plt.plot(out_firing_times, out_offsets, '.')
    #plt.show()

    plot_delays_variances(out.delayst, out.variancest)
    #standard_plots(out.spike_time_trace, out.iapp_trace, out.vt, output=3)

    print('end')



if __name__ == '__main__':
    #main()
    plot_simple_sdvl_example()