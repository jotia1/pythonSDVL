import numpy as np
MSPERSEC = 1000


# Might be useful one day
# num_presentations = 10
# idxs = np.tile([0, 1, 2], num_presentations)
# ts = p_ts = np.reshape(np.tile(np.arange(0, num_presentations), (3, 1)), (-1), order='F') * 500 + np.tile([1, 2, 3], num_presentations)


def test_case(N_inp):
    return np.array([0, 1, 2, 0, 1, 2]), np.array([1, 3, 7, 501, 503, 507]), [0, 500]

def random_data(N_inp, sim_time_sec):
    inp_idxs = np.random.randint(0, N_inp, (N_inp + 1) * 10 * sim_time_sec)
    inp_ts = np.random.randint(0, sim_time_sec * MSPERSEC, (N_inp + 1) * 10 * sim_time_sec)
    return inp_idxs, inp_ts, []

def embedded_pattern(Tp, Df, N_inp, Np, Pf, p_inp, p_ts, p_fun, dropout):
    exp_spikes_in_presentation = round(Np *( 1 - dropout))

    # Calculate base frequencies
    non_patt_time_sec = (MSPERSEC - (Pf * Tp)) / MSPERSEC
    bottom_base_freq = (Df - (Pf * (1 - dropout))) / non_patt_time_sec

    exp_N_inp_spikes_Tp = np.ceil(Df * Tp / MSPERSEC * N_inp)
    exp_N_dist_spikes_Tp = exp_N_inp_spikes_Tp - exp_spikes_in_presentation
    exp_total_N_dist_spikes_Tp = exp_N_dist_spikes_Tp * Pf

    N_dist = N_inp - Np
    exp_N_dist_total_spikes = Df * N_dist
    exp_N_dist_total_spikes_npt = exp_N_dist_total_spikes - exp_total_N_dist_spikes_Tp
    N_dist_npt_freq = exp_N_dist_total_spikes_npt / N_dist / non_patt_time_sec

    # Generate base spike trains
    toppoiss = np.random.poisson(N_dist_npt_freq * N_dist)
    top_inp = np.random.randint(Np, N_inp + 1, (1, toppoiss))
    top_ts = np.random.randint(0, MSPERSEC, (1, toppoiss))

    bottpoiss = np.random.poisson(Np * bottom_base_freq)
    bott_inp = np.random.randint(0, Np, (1, bottpoiss))
    bott_ts = np.random.randint(0, MSPERSEC, (1, bottpoiss))

    inp = np.concatenate((top_inp, bott_inp), axis=1)
    ts = np.concatenate((top_ts, bott_ts), axis=1) 

    # Add pattern
    offsets = []
    max_delay = 20
    slot_size = Tp + max_delay
    num_slots = int(np.floor(MSPERSEC / slot_size))
    probability = Pf / num_slots
    selected_slots = np.where(np.random.rand(num_slots) < probability)[0]

    left_over_time = (MSPERSEC % slot_size)
    patt_start_time_offset = np.random.randint(0, left_over_time)

    for i in selected_slots:
        offset = i * slot_size + patt_start_time_offset
        
        spike_filter = np.where(np.logical_and(ts > offset, ts <= offset + Tp))
        ts = np.delete(ts, spike_filter)
        inp = np.delete(inp, spike_filter)

        # Generate distractor spikes simultaneous to pattern
        offsetpoiss = np.random.poisson(exp_N_dist_spikes_Tp)
        offset_inp = np.random.randint(Np, N_inp + 1, (offsetpoiss))
        offset_ts = np.random.randint(0, Tp + 1, (offsetpoiss))

        patt_toinsert = p_inp
        ts_toinsert = p_ts

        inp = np.concatenate((inp, patt_toinsert, offset_inp))
        ts = np.concatenate((ts, ts_toinsert + offset, offset_ts + offset))
        offsets.append(offset)
    
    return inp, ts, offsets

#p_inp = np.arange(0, 500)
#p_ts = np.reshape(np.tile(np.arange(0, 50), (10, 1)), (-1), order='F')
#embedded_pattern(50, 10, 2000, 500, 5, p_inp, p_ts, None, 0.0)