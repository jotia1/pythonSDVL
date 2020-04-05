import numpy as np
from abc import ABC, abstractmethod 

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