from simtools import *
import numpy as np

def trueposxtrueneg(net, out, sim_params):
    testing_seconds = sim_params.test_seconds
    training_ms = (sim_params.sim_time_sec - testing_seconds) * MSPERSEC

    output_filter = np.logical_and(out.spike_time_trace[:, 1] == net.N-1, 
            out.spike_time_trace[:, 0] > training_ms)
    output_spike_times = out.spike_time_trace[output_filter, 0]

    test_offsets = out.offsets[out.offsets >= training_ms]

    last_offset = -sim_params.Tp - net.delay_max

    true_positives = 0
    true_negatives = 0
    num_true_neg_slots = 0
    for offset in test_offsets:

        # Any spikes since end of last pattern
        end_last_offset = last_offset + sim_params.Tp + net.delay_max
        pre_offset_spikes = np.any(np.logical_and(
                        output_spike_times >= end_last_offset,
                        output_spike_times <= offset))
        
        # Count previous gap as a true neg slot and whether it was a true neg
        if offset > end_last_offset:
            num_true_neg_slots += 1
            if not pre_offset_spikes:
                true_negatives += 1
        
        last_offset = offset

        # Now this pattern offset
        offset_spikes = np.any(np.logical_and(
                    output_spike_times > offset,
                    output_spike_times < offset + sim_params.Tp + net.delay_max))
        if offset_spikes:
            true_positives += 1

    if len(test_offsets) == 0 or num_true_neg_slots == 0:
        return 0
    prop_TP = true_positives / len(test_offsets)
    prop_TN = true_negatives / num_true_neg_slots

    return prop_TP * prop_TN