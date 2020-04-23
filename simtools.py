import numpy as np
import logging
import time as timer
from datasource import embedded_pattern, test_case
from networks import *
import json

MSPERSEC = 1000
COMBINEFLAG='--combine'

# Types of input data that can be used
STANDARDINPUT='STANDARDINPUT'
TESTINPUT='TESTINPUT'

ALLSIMPARAMS = {
    'sim_time_sec',
    'time_execution',
    'Tp',
    'Df',
    'Pf',
    'naf',
    'test_seconds',  
    'p_inp',
    'p_ts',
    'inp_idxs',
    'inp_ts',
    'inp_type',
    'data_fcn',
    'voltages_to_save',
    'delays_to_save',
    'variances_to_save',  
    'net_params_dict',
    'output_folder',
    'output_base_filename',
}

class SimulationTimer():

    def __init__(self, time_execution=False):
        """ Create a SimulationTimer

        Optionally can choose to time a simulation or not, this logging
        to stay in the code without having using memory / time for logging.

        time_execution (bool) : Whether or not to time this execution of the program or not
        """
        self.times = {}
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





class OldSimulationParameters():
    def __init__(self, exp_params, slurm_id=1, task_id=1):
        assert self.validate_params(exp_params), 'exp_params not valid'
        self.exp_params = exp_params
        self.slurm_id = slurm_id
        self.task_id = task_id
        for key, value in exp_params.items():
            setattr(self, key, value)

        self.input_provided = False
        if self.inp_idxs and self.inp_ts:
            print('Input data provided.')
            self.input_provided = True
            self.inp_idxs = np.array(self.inp_idxs)
            self.inp_ts = np.array(self.inp_ts)

            if not self.data_fcn:
                self.data_fcn = lambda :self.inp_idxs, self.inp_ts, [0]
            else:
                print('Both input data and data_fcn provided. This is odd.')

        # If there is no input and no datafcn provided make one
        if not self.input_provided and not self.data_fcn:
            if not self.p_inp or not self.p_ts: # No pattern, make one
                self.p_inp = np.arange(0, 500)
                self.p_ts = np.reshape(np.tile(np.arange(0, self.Tp), (10, 1)), (-1), order='F')
            
            if self.inp_type == STANDARDINPUT:
                self.data_fcn = lambda n_inp : embedded_pattern(self.Tp, self.Df, n_inp, self.naf, self.Pf, self.p_inp, self.p_ts, None, 0.0)
            if self.inp_type == TESTINPUT:
                self.data_fcn = test_case
            #else:
            #    raise Exception(f'Unknown inp_type: {self.inp_type}. Cannot create input for simulation.')
        
        self.voltages_to_save = np.array([] if not self.voltages_to_save else self.voltages_to_save, dtype=np.int32) 
        self.delays_to_save = np.array([] if not self.delays_to_save else self.delays_to_save, dtype=np.int32)
        self.variances_to_save = np.array([] if not self.variances_to_save else self.variances_to_save, dtype=np.int32)

    

    @property
    def output_folder(self):
        return f'{self.job_name}_{self.slurm_id}'

    @property
    def output_base_filename(self):
        return f'{self.output_folder}_{self.task_id}'

    @property
    def full_filepath(self):
        return f'{self.output_folder}/{self.output_base_filename}'

    def validate_params(self, exp_params):
        return True # TODO : Verify required params exist


class SimulationOuput():
    def __init__(self):
        self.sim_timer = None
        self.vt = None
        self.variancest=None
        self.delayst=None
        self.iapp_trace = None
        self.spike_time_trace = None

class ParametersObject():
    _all_params = set()
    def __init__(self, from_dict):
        self.from_dict = from_dict
        if from_dict:
            self.load_from_dict(from_dict)

    def load_from_dict(self, from_dict):
        missing = self.missing_params(from_dict)
        if len(missing) != 0:
            raise Exception(f"Parameters missing: {missing}")

        if not self.is_valid(from_dict):
            raise Exception("From dict not valid.")

        for key, value in from_dict.items():
            setattr(self, key, value)

    def missing_params(self, from_dict):
        missing = []
        for param in self._all_params:
            if param not in from_dict.keys():
                missing.append(param)
        return missing

    def to_dict(self):
        params_dict = {}
        for key in self._all_params:
            params_dict[key] = getattr(self, key)
        return params_dict

    def is_valid(self, from_dict):
        """ Returns True iff the given dict has a key for each value
            required for a Parameter Object
        """
        return from_dict != None and self.missing_params(from_dict) == []


class SimulationParameters(ParametersObject):
    _all_params = ALLSIMPARAMS
    def __init__(self, from_dict):
        # For the sake of linting...
        self.sim_time_sec = None
        self.time_execution  = None
        self.Tp = None
        self.Df = None
        self.Pf = None
        self.naf = None
        self.test_seconds = None
        self.p_inp = None
        self.p_ts = None
        self.inp_idxs = None
        self.inp_ts = None
        self.inp_type = None
        self.data_fcn = None
        self.voltages_to_save = None
        self.delays_to_save = None
        self.variances_to_save  = None
        self.net_params_dict = None
        self.output_folder = None
        self.output_base_filename = None

        super().__init__(from_dict)

        self.voltages_to_save = np.array([] if not self.voltages_to_save else self.voltages_to_save, dtype=np.int32) 
        self.delays_to_save = np.array([] if not self.delays_to_save else self.delays_to_save, dtype=np.int32)
        self.variances_to_save = np.array([] if not self.variances_to_save else self.variances_to_save, dtype=np.int32)


        # self.voltages_to_save = np.array(self.voltages_to_save)
        # self.delays_to_save = np.array(self.delays_to_save)
        # self.variances_to_save = np.array(self.variances_to_save)

        self.setup_input()

    def setup_input(self):
        # TODO : Make dynamic and general
        print('USING HARDCODED DATA FUNCTION')
        self.input_provided = False

        if self.inp_type == STANDARDINPUT:
            self.p_inp = np.arange(0, 500)
            self.p_ts = np.reshape(np.tile(np.arange(0, self.Tp), (10, 1)), (-1), order='F')
            self.data_fcn = lambda n_inp : embedded_pattern(self.Tp, self.Df, n_inp, self.naf, self.Pf, self.p_inp, self.p_ts, None, 0.0)
        elif self.inp_type == TESTINPUT:
            self.data_fcn = test_case

    def build_network(self):
        return Network(self.net_params_dict)

    def to_dict(self):
        sim_params_dict = super().to_dict()
        sim_params_dict['data_fcn'] = None
        sim_params_dict['voltages_to_save'] = self.voltages_to_save.to_list()
        sim_params_dict['delays_to_save'] = self.delays_to_save.to_list()
        sim_params_dict['variances_to_save'] = self.variances_to_save.to_list()
        return sim_params_dict
        

class NetworkParameters(ParametersObject):
    _all_params = set()  # TODO if we use this class
    def __init__(self, from_dict):
        self.group_sizes = None
        self.N_inp = None
        self.N = None

        self.delay_init = None
        self.variance_init = None
        self.w_init = None
        self.connections = None

        self.fgi = None

        self.v_rest = None
        self.v_reset = None
        self.v_threshold = None
        self.neuron_tau = None
        self.w_max = None
        
        self.varaince_min = None
        self.variance_max = None
        self.delay_max = None

        self.a1 = None
        self.a2 = None
        self.b1 = None
        self.b2 = None
        self.nu = None
        self.nv = None

        super().__init__(from_dict)


def save_sim_params(filename, sim_params):
    sim_params_dict = sim_params.to_dict()
    save_dict(filename, sim_params_dict)

def save_dict(filename, data):
    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)

def load_dict(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def load_sim_params(filename):
    sim_params_dict = load_dict(filename)
    return SimulationParameters(sim_params_dict)


def save_experiment(net, out, sim_params, only_output=True):
    # spike_time_trace, vt, iapp, offsets, 
    np.savez(f'{sim_params.output_folder}/{sim_params.output_base_filename}.npz', 
                                    spike_time_trace=out.spike_time_trace[out.spike_time_trace[:, 1] == net.N-1, :] if only_output else out.spike_time_trace,
                                    vt=out.vt,
                                    result=out.result,
                                    iapp_trace=out.iapp_trace,
                                    offsets=out.offsets,
                                    variancest=out.variancest,
                                    delayst=out.delayst)

def load_experiment(filename):
    npz = np.load(filename)
    out = SimulationOuput()
    for key,value in npz.items():
        setattr(out, key, value)
    return out
    # out.spike_time_trace = npz['spike_time_trace']
    # out.vt = npz['vt']
    # out.result = npz['result']
    # out.iapp_trace = npz['iapp_trace']
    # out.offsets = npz['offsets']
    # out.variancest = npz['variancest']
    # out.dealyst = npz['delayst']
    # return out



def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger