import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
sns.set()


def standard_plots_ipython(out, output=2000):
    ax = plt.subplot(311)
    plt.gcf().set_size_inches(26, 8*3, forward=True)
    plot_spike_times(out.spike_time_trace, output=output, axg=ax)
    overlay_offsets(out.offsets)
    ax = plt.subplot(312)
    plt.gcf().set_size_inches(26, 8*3, forward=True)
    plot_trace(np.array(out.iapp_trace), axg=ax)
    ax.set_title('Current to output neuron')
    overlay_offsets(out.offsets)
    ax = plt.subplot(313)
    plt.gcf().set_size_inches(26, 8*3, forward=True)
    plot_trace(out.vt[0, :], axg=ax)
    ax.set_title('Voltage trace')
    overlay_offsets(out.offsets)
    plt.tight_layout()
    plt.show()

def standard_plots(out, output=2000):
    ax = plt.subplot(211)
    plot_spike_times(out.spike_time_trace, output=output, axg=ax)
    overlay_offsets(out.offsets)
    ax = plt.subplot(223)
    plot_trace(np.array(out.iapp_trace), axg=ax)
    ax.set_title('Current to output neuron')
    overlay_offsets(out.offsets)
    ax = plt.subplot(224)
    plot_trace(out.vt[0, :], axg=ax)
    ax.set_title('Voltage trace')
    overlay_offsets(out.offsets)
    plt.tight_layout()
    plt.show()

def overlay_offsets(offsets, axg=None):
    ax = axg if axg else plt.gca()
    rects = []
    ybot, ytop = ax.get_ylim()
    for offset in offsets:
        rect = plt.Rectangle((offset, ybot), 70, ytop - ybot)
        rects.append(rect)
    pc = PatchCollection(rects, alpha=0.3)
    ax.add_collection(pc)

def plot_delays_variances(delayst, variancest, num_to_plot=3):
    ax0=plt.subplot(211)
    for i in range(num_to_plot):
        ax0.plot(delayst[i, 0, :])
    ax1=plt.subplot(212)
    ax0.set_title('Changes in delay and variance')
    #ax0.yaxis.tick_right()
    #ax0.set_xlabel('Simulation time (sec)')
    ax0.xaxis.set_visible(False)
    ax0.set_ylabel('Delay (ms)')
    for i in range(num_to_plot):
        ax1.plot(variancest[i, 0, :] + 0.0 * i - 0.0, label=f'Input #{i+1}')
    ax0.set_facecolor((0,0,0,0))
    ax1.set_facecolor((0,0,0,0))
    #ax1.set_title('Variances')
    ax1.set_xticklabels([str(x) for x in range(-5, 31, 5)])
    ax1.set_xlabel('Training time (sec)')
    ax1.set_ylabel('Variance')
    ax1.legend()
    plt.show()

def iplot(fnc, *args, **kwargs):
    ax = plt.gca()
    plt.gcf().set_size_inches(26, 8, forward=True)
    fnc(*args, **kwargs)
    #overlay_offsets(out.offsets, axg=ax)

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
    return axg

def plot_trace(trace, axg=None):
    
    ax = axg if axg else plt.gca()

    ax.plot(list(range(trace.size)), trace)

    None if axg else plt.show()
    return axg