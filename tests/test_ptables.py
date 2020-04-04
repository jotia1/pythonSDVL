import sys, os
testdir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(testdir, '../')))

import numpy as np
from mvp import *


def test_ptable_shape():
    variance_precision = 0.01
    var_range = np.concatenate((np.arange(0.1, 10, variance_precision), [10]))
    ptable = getlookuptable(var_range,
                            np.arange(0, 20),
                            np.arange(0, 40),
                            13)

    assert ptable.shape == (991, 20, 40)

def test_ptable_error_margin():
    variance_precision = 0.01
    var_range = np.concatenate((np.arange(0.1, 10, variance_precision), [10]))
    ptable = getlookuptable(var_range,
                            np.arange(0, 20),
                            np.arange(0, 40),
                            13)

    # TODO
    assert False