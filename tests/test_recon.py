from unittest import TestCase, mock
import os

import numpy as np
import h5py

import tomopy
from tomopy_cli.recon import rec


HDF_FILE = './test_tomogram.h5'


def make_params():
    params = mock.MagicMock()
    params.file_name = HDF_FILE
    params.rotation_axis = 32
    params.file_type = 'standard'
    params.file_format = 'dx'
    params.flat_correction_method = 'standard'
    params.reconstruction_algorithm = 'gridrec'
    params.gridrec_filter = 'parzen'
    params.reconstruction_mask_ratio = 1.0
    params.reconstruction_type = 'slice'
    return params


class ReconTests(TestCase):
    def setUp(self):
        # Prepare some dummy data
        phantom = tomopy.misc.phantom.shepp3d(size=64)
        phantom = np.exp(-phantom)
        flat = np.ones((2, *phantom.shape[1:]))
        dark = np.zeros((2, *phantom.shape[1:]))
        with h5py.File(HDF_FILE, mode='w-') as fp:
            fp.create_dataset('/exchange/data', data=phantom)
            fp.create_dataset('/exchange/data_white', data=flat)
            fp.create_dataset('/exchange/data_dark', data=dark)
    
    def tearDown(self):
        if os.path.exists(HDF_FILE):
            os.remove(HDF_FILE)
        
    def test_basic_reconstruction(self):
        params = make_params()
        params.reconstruction_type = 'slice'
        response = rec(params=params)
