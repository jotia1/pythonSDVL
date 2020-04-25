import numpy as np
from abc import ABC, abstractmethod 

class Network(object):

    def __init__(self, net_params_dict):
        if not self.valid_params(net_params_dict):
            raise Exception("Network Parameters not valid")

        self.group_sizes = None
        self.N_inp = None
        self.N = None
        
        self.delays = None
        self.delay_init = None
        self.variance = None
        self.variance_init = None
        self.w = None
        self.w_init = None
        self.connections = None

        self.fgi = None

        self.v_rest = None
        self.v_reset = None
        self.v_threshold = None
        self.neuron_tau = None
        self.w_max = None

        self.variance_min = None
        self.variance_max = None
        self.delay_max = None

        self.a1 = None
        self.a2 = None
        self.b1 = None
        self.b2 = None
        self.nu = None
        self.nv = None
        
        for key, value in net_params_dict.items():
            setattr(self, key, value)

        # Set up connectivity
        # TODO : Set up delays/variances/w based on connections, not reverse
        self.delays = np.zeros([self.N, self.N])
        self.delays[0:self.N_inp, self.N_inp:] = self.delay_init
        self.variance = np.zeros([self.N, self.N])
        self.variance[0:self.N_inp, self.N_inp:] = self.variance_init
        self.w = np.zeros([self.N, self.N])
        self.w[0:self.N_inp, self.N_inp:] = self.w_init
        self.connections = np.nonzero(self.w)
        


    def valid_params(self, net_params):
        return net_params != None

    # def __init__(self, n_inputs=2000):
        # self.group_sizes = np.array([n_inputs, 1])
        # self.N_inp = self.group_sizes[0]
        # self.N = np.sum(self.group_sizes)
        
        # self.delays = np.zeros([self.N, self.N])
        # self.delays[0:self.N_inp, self.N_inp:] = 5
        # self.variance = np.zeros([self.N, self.N])
        # self.variance[0:self.N_inp, self.N_inp:] = 2
        # self.w = np.zeros([self.N, self.N])
        # self.w[0:self.N_inp, self.N_inp:] = 1
        # self.connections = np.nonzero(self.w)

        # self.fgi = 0.0226

        # self.v_rest = -65
        # self.v_reset = -70
        # self.v_threshold = -55
        # self.neuron_tau = 20
        # self.w_max = 10

        # self.variance_min = 0.1
        # self.variance_max = 10
        # self.delay_max = 20

        # self.a1 = 3
        # self.a2 = 20
        # self.b1 = 20
        # self.b2 = 20
        # self.nu = 0.04
        # self.nv = 0.04