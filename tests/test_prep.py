import unittest
from unittest import mock
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from tomopy_cli import prep

class PrepTests(unittest.TestCase):
    '''Class to test tomopy_cli/prep.py
    '''
    def test_minus_log(self):
        params = mock.MagicMock()
        params.minus_log = False
        data = np.linspace(0.05, 1.0, 20)
        np.testing.assert_array_equal(prep.minus_log(data, params), data)
        params.minus_log = True
        ext_length_data = prep.minus_log(data, params)
        self.assertAlmostEqual(ext_length_data[-1], 0.0)
        self.assertAlmostEqual(ext_length_data[9], 0.693147181)

if __name__ == '__main__':
    unittest.main()
