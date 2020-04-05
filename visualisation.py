import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def standard_plots(spike_time_trace, iapp_trace, vt):
    ax = plt.subplot(211)
    plot_spike_times(spike_time_trace, output=2000, axg=ax)
    ax = plt.subplot(223)
    plot_trace(np.array(iapp_trace), axg=ax)
    ax.set_title('Current to output neuron')
    ax = plt.subplot(224)
    plot_trace(vt[0, :], axg=ax)
    ax.set_title('Voltage trace')
    plt.show()


def plot_spike_times(data, output=None, axg=None):
    """ Takes the spike_time_trace variable
    ndarry with shape (-1, 2) where data[:,0] are the times and
    data[:, 1] are the spike indexs.
    Only one output neuron is currently supported, output should be an int
    Optinally takes axes to draw in else creates, if axg is given, plt.show is not called.
    """
    ax = axg if axg else plt.gca()

    # May be useful later, creates a transparent polygon (for highlighting pattern?)
    #    ax.add_patch(
    #    plt.Polygon(poly_coords, color='forestgreen', alpha=0.5)

    ax.plot(data[:, 0], data[:, 1], '.')

    if output:
        #orig_indices = data[:, 1].argsort()
        #output_spikes =  orig_indices[np.searchsorted(data[orig_indices, 1], np.array(outputs))]
        output_spikes = data[:, 1] == output
        ax.plot(data[output_spikes, 0], data[output_spikes, 1], '.r')

    None if axg else plt.show()

def plot_trace(trace, axg=None):
    
    ax = axg if axg else plt.gca()

    ax.plot(list(range(trace.size)), trace)

    None if axg else plt.show()