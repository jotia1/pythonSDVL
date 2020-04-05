import numpy as np
import time as timer
import simtools as st

MSPERSEC = 1000
PTABLEDELAYINDEXOFFSET = 1
PTABLEVARIANCEINDEXOFFSET = 10
DELAYMIN = 1
TIMEEXECUTION = True

def simulate(net, sim_params):
    logger = st.setup_logging(f"{__name__}.simulate")

    sim_time_sec = sim_params.sim_time_sec
    sim_time_ms = sim_time_sec * MSPERSEC
    
    inp_idxs = np.random.randint(0, net.N-1, net.N * 10 * sim_time_sec)
    inp_ts = np.random.randint(0, sim_time_sec * MSPERSEC, net.N * 10 * sim_time_sec)

    current_steps = 40
    upcoming_current = np.zeros([net.N, current_steps])
    upcur_idx = 0

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
    sim_timer = st.SimulationTimer(TIMEEXECUTION)
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

        sim_timer.log_time(time)                                # 10 

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
        
        sim_timer.log_time(time)                                # 10 
        
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

            sim_timer.log_time(time)                        # 25 %

            # Update SDVL means
            du = np.zeros(t0_negu.shape)
            du[t0 >= net.a2] = -k[t0 >= net.a2] * net.nu
            du[abs_t0_negu >= net.a1] = shifts[abs_t0_negu >= net.a1]

            sim_timer.log_time(time)                        # 9%

            net.delays[:, fired] += du
            net.delays[net.connections] = np.maximum(DELAYMIN, 
                                        np.minimum(net.delay_max, 
                                        net.delays[net.connections]))
            
            sim_timer.log_time(time)                        # 17 %

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

        sim_timer.log_time(time)                           # 25 %

        #if time % MSPERSEC == 0:
        #    big_stt.append(spike_time_trace)
        #    spike_time_trace = np.empty((0, 2))

        sim_timer.log_time(time)
    
    print('Time taken: ', timer.time() - start_time)
    logger.info('Simulation finished')

    #  HACK : remove
    net.vt = vt
    net.iapp_trace = iapp_trace
    net.spike_time_trace = spike_time_trace

    return net, sim_timer

class SimulationParameters():
    def __init__(self):
        self.sim_time_sec = 2

def getlookuptable(var_range, delays_range, steps_range, fgi):
    """ Table for postsynaptic currents for given delay and variance

    Build a 3D table, where rows are a range of variances, columns are a
    range of delays and depths are an amount of current to deliver at that
    time step for the given delay and variance.
    """
    logger = st.setup_logging(f'{__name__}.getlookuptable')
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
