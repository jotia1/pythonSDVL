import numpy as np
import logging
import time as timer

MSPERSEC = 1000
COMBINEFLAG='--combine'

# Types of input data that can be used
STANDARDINPUT='STANDARDINPUT'

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


def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger