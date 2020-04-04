import numpy as np
import logging
import seaborn as sns
import time as timer
import matplotlib.pyplot as plt
sns.set()

MSPERSEC = 1000
PTABLEDELAYINDEXOFFSET = 1
PTABLEVARIANCEINDEXOFFSET = 10
DELAYMIN = 1
TIMEEXECUTION = False


class Network(object):
    def __init__(self):
        self.group_sizes = np.array([2000, 1])
        self.N_inp = self.group_sizes[0]
        self.N = np.sum(self.group_sizes)
        
        self.delays = np.zeros([self.N, self.N])
        self.delays[0:self.N_inp, self.N_inp:] = 5
        self.variance = np.zeros([self.N, self.N])
        self.variance[0:self.N_inp, self.N_inp:] = 2
        self.w = np.zeros([self.N, self.N])
        self.w[0:self.N_inp, self.N_inp:] = 1
        self.connections = np.nonzero(self.w)

        self.fgi = 0.0230

        self.v_rest = -65
        self.v_reset = -70
        self.v_threshold = -55
        self.neuron_tau = 20
        self.w_max = 10

        self.variance_min = 0.1
        self.variance_max = 10
        self.delay_max = 20
        self.voltages_to_save = np.array([self.N-1])

        self.test_seconds = 0

        self.a1 = 3
        self.a2 = 20
        self.b1 = 20
        self.b2 = 20
        self.nu = 0.04
        self.nv = 0.04
        


def main():
    logger = setup_logging(f"{__name__}.main")

    sim_time_sec = 2
    sim_time_ms = sim_time_sec * MSPERSEC
    net = Network()
    
    #inp_idxs = np.array([2, 1, 0, 0])
    #inp_ts = np.array([1, 3, 5, 10])
    inp_idxs = np.random.randint(0, net.N-1, net.N * 10)
    inp_ts = np.random.randint(0, sim_time_sec * MSPERSEC, net.N * 10)

    current_steps = 40
    upcoming_current = np.zeros([net.N, current_steps])
    upcur_idx = 0

    voltages_to_save = np.array([3])
    v = np.ones(net.N) * net.v_rest
    vt = np.zeros([len(net.voltages_to_save), sim_time_ms])

    spike_time_trace = np.empty((0, 2))
    last_spike_time = np.zeros((net.N, 1)) * -np.Inf
    big_stt = []  # For sake of time comparison to matlab implementation

    # Get a table of gaussians
    variance_precision = 0.01
    var_range = np.concatenate((np.arange(net.variance_min, net.variance_max, variance_precision), [10]))
    ptable = getlookuptable(var_range,
                            np.arange(0, net.delay_max),
                            np.arange(0, current_steps),
                            net.fgi)

    iapp_trace = []
    fn_trace = []
    fired_trace = []

    start_time = timer.time()
    sim_timer = SimulationTimer(TIMEEXECUTION)
    for time in range(sim_time_ms):
        #logger.debug(f'Timestep: {time} ms')

        sim_timer.log_time(time)

        Iapp = upcoming_current[:, upcur_idx]
        iapp_trace.append(Iapp[net.N-1])

        # Update membrane equations
        v += ((net.v_rest - v) / net.neuron_tau) + Iapp
        vt[:, time] = v[net.voltages_to_save]

        fired_naturally = np.where(v > net.v_threshold)[0]
        fired_inputs = inp_idxs[inp_ts == time]
        fired = np.concatenate((fired_naturally, fired_inputs))
        fired_spike_times = np.concatenate((time * np.ones(fired.shape).reshape((-1, 1)), fired.reshape((-1, 1))), axis=1)
        spike_time_trace = np.concatenate((spike_time_trace, 
                fired_spike_times), axis=0)
        last_spike_time[fired] = time

        fn_trace.extend(fired_naturally.tolist())
        fired_trace.extend(zip(fired.tolist(), [time for x in range(fired.size)]))

        sim_timer.log_time(time)

        # Get all possible post-synaptic connections
        fired_delays = np.round(net.delays[fired, :])
        fired_var = net.variance[fired, :]
        fired_w = net.w[fired, :]

        # Prune to only connections that actually exist (have a delay)
        fired_conns = np.nonzero(fired_delays)

        # Grab the actual values for delays, variance and w for each connection
        conn_delays = fired_delays[fired_conns]
        conn_var = fired_var[fired_conns]
        conn_w = fired_w[fired_conns]

        # Convert delays and variances into values that can be used to
        # index into ptables
        # Adjust for offsets in min value and decimals
        indexs_delays = conn_delays - PTABLEDELAYINDEXOFFSET
        indexs_var = np.round(conn_var / variance_precision - PTABLEVARIANCEINDEXOFFSET)

        p_values = ptable[indexs_var.astype(int), indexs_delays.astype(int), :]
        stepped_current = np.zeros(p_values.shape)
        if conn_w.size > 0:
            stepped_current = conn_w.reshape((conn_w.size, 1)) * p_values
        
        sim_timer.log_time(time) 
        
        # TODO This is a hack for a single neuron.
        # For arbitrary second layers will need to be more careful about how we sum...
        output_currents = np.sum(stepped_current, axis=0)

        # Now lets continue pretending we have multiple rows 
        weighted_gauss_samples = np.zeros((net.N, current_steps))
        weighted_gauss_samples[-1, :] = output_currents
        
        sim_timer.log_time(time) 

        # Figure out what the upcoming currents are
        upcoming_current[:, upcur_idx] = 0
        upcur_idx = (upcur_idx + 1) % current_steps
        idx_diff = - upcur_idx
        upcoming_current[:, upcur_idx:] += weighted_gauss_samples[:, :idx_diff or None]
        upcoming_current[:, :upcur_idx or None] += weighted_gauss_samples[:, idx_diff:]
    
        # Reset any neurons that have fired
        v[fired] = net.v_reset

        sim_timer.log_time(time) 

        ###     LEARNING

        ##      STDP
        # TODO : whenever it seems relevent... 

        # Bound weights
        #net.w = np.maximum(0, np.minimum(net.w_max, net.w))

        sim_timer.log_time(time) 
        ##      SDVL
        # Do not adjust synapses during testing
        if time < ((sim_time_sec - net.test_seconds) * MSPERSEC):

            sim_timer.log_time(time)

            t0 = np.broadcast_to(time - last_spike_time, (net.N, fired.size))
            t0_negu = t0 - net.delays[:, fired]
            abs_t0_negu = np.abs(t0_negu)
            k = np.power(net.variance[:, fired], 2) 
            shifts = np.sign(t0_negu) * k * net.nu

            sim_timer.log_time(time)

            # Update SDVL means
            du = np.zeros(t0_negu.shape)
            du[t0 >= net.a2] = -k[t0 >= net.a2] * net.nu
            du[abs_t0_negu >= net.a1] = shifts[abs_t0_negu >= net.a1]

            sim_timer.log_time(time) 

            net.delays[:, fired] += du
            net.delays[net.connections] = np.maximum(DELAYMIN, 
                                        np.minimum(net.delay_max, 
                                        net.delays[net.connections]))
            
            sim_timer.log_time(time) 

            # Update SDVL variance
            dvar = np.zeros(t0_negu.shape)
            dvar[abs_t0_negu <= net.b2] = -k[abs_t0_negu <= net.b2] * net.nv
            dvar[abs_t0_negu >= net.b1] = k[abs_t0_negu >= net.b1] * net.nv

            net.variance[:, fired] += dvar
            net.variance[net.connections] = np.maximum(net.variance_min, 
                                            np.minimum(net.variance_max, 
                                            net.variance[net.connections]))

            # TODO : investigate the variance net.connections. Could it be used above when 
            # trying calculate the upcoming_currents

        sim_timer.log_time(time)

        if time % MSPERSEC == 0:
            big_stt.append(spike_time_trace)
            spike_time_trace = np.empty((0, 2))

        sim_timer.log_time(time)
    
    print('Time taken: ', timer.time() - start_time)
    #print(sim_timer.get_percentages())
    #standard_plots(spike_time_trace, iapp_trace, vt)
    logger.info('Simulation finished')
    

class SimulationTimer():

    times = {}

    def __init__(self, time_execution=False):
        """ Create a SimulationTimer

        Optionally can choose to time a simulation or not, this logging
        to stay in the code without having using memory / time for logging.

        time_execution (bool) : Whether or not to time this execution of the program or not
        """
        self.time_execution = time_execution  

    def log_time(self, time):
        """ Helper function for timing execution

        Pass in the current millisecond and it will be recorded in the global

        time (int) : The current ms of the simulation
        """ 
        if not self.time_execution:
            return

        time_records = self.times.get(time, [])
        time_records.append(timer.time())
        self.times[time] = time_records

    def get_times(self):
        """ Return the times recorded

        Each col is ordered based on the order of calls to log_time with a specific time
        The rows are NOT in order of time
        """
        return np.array(list(self.times.values()))

    def get_percentages(self):
        if not self.time_execution:
            return np.array([])
        data = self.get_times()
        sums = np.sum(data, axis=0)
        section_times = sums[1:] - sums[:-1] 
        return np.round(section_times / np.sum(section_times) * 100) 

    def reset(self):
        self.times = {}

def standard_plots(spike_time_trace, iapp_trace, vt):
    ax = plt.subplot(211)
    plot_spike_times(spike_time_trace, output=2000, axg=ax)
    ax = plt.subplot(223)
    plot_trace(np.array(iapp_trace), axg=ax)
    ax.title('Current to output neuron')
    ax = plt.subplot(224)
    plot_trace(vt[0, :], axg=ax)
    ax.title('Voltage trace')
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

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def plot_trace(trace, axg=None):

    ax = axg if axg else plt.gca()

    ax.plot(list(range(trace.size)), trace)

    None if axg else plt.show()

def getlookuptable(var_range, delays_range, steps_range, fgi):
    """ Table for postsynaptic currents for given delay and variance

    Build a 3D table, where rows are a range of variances, columns are a
    range of delays and depths are an amount of current to deliver at that
    time step for the given delay and variance.
    """
    logger = setup_logging(f'{__name__}.getlookuptable')
    accuracy = 0.001
    ptable_filename = f"ptables/ptable_{str(fgi).replace('.', '')}_{str(accuracy).replace('.', '')}.npy"
    
    try:
        ptable = np.load(ptable_filename)
        logger.debug(f"Loaded: {ptable_filename}")
        return ptable
    except IOError:
        logger.debug(f'Failed to load: {ptable_filename}')

    logger.info('Building new lookup table.')
    steps = np.tile(steps_range.reshape((1, 1, -1)), (var_range.size, delays_range.size, 1))
    delays = np.tile(delays_range.reshape((1, -1, 1)), (var_range.size, 1, steps_range.size))
    variances = np.tile(var_range.reshape((-1, 1, 1)), (1, delays_range.size, steps_range.size))

    exptable = np.exp(- np.power(steps - delays, 2) / (2 * variances) )
    sample_delays = -np.round(delays[:, :, 1])
    sample_variances = variances[:, :, 1]

    #ptable = np.zeros((len(var_range), len(delays_range), len(steps_range)))
    p = np.full(sample_variances.shape, fgi) / np.sqrt(2 * np.pi * sample_variances)

    max_error = fgi * accuracy
    adjustment_term = fgi * accuracy * 0.05
    small_peaks = 0
    big_peaks = 0
    do = True
    while do:
        p += adjustment_term * small_peaks
        p -= adjustment_term * big_peaks

        full_integral = np.zeros(sample_delays.shape)
        for j in range(steps_range.size):
            step_integral = p * np.exp(- np.power(sample_delays + j, 2) / (2 * sample_variances) )
            full_integral += step_integral

        small_peaks = full_integral < (fgi - max_error)
        big_peaks = full_integral > (fgi + max_error)
        do = np.any(small_peaks) or np.any(big_peaks)
    
    ptable = np.repeat(np.reshape(p, (var_range.size, delays_range.size, 1)), 40, 2) * exptable

    try:
        np.save(ptable_filename, ptable)
        logger.debug(f'Saved: {ptable_filename}')
    except Exception:
        logger.debug(f'Failed to save: {ptable_filename}')

    return ptable

if __name__ == '__main__':
    main()