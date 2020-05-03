from unittest import TestCase, mock
import os
import shutil
from pathlib import Path

import numpy as np
import h5py

import tomopy
from tomopy_cli.recon import rec

HDF_FILE = Path(__file__).resolve().parent / 'test_tomogram.h5'


def make_params():
    params = mock.MagicMock()
    params.file_name = str(HDF_FILE)
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
    output_dir = Path(__file__).resolve().parent.parent / 'tests_rec'

    def setUp(self):
        # Prepare some dummy data
        phantom = tomopy.misc.phantom.shepp3d(size=64)
        phantom = np.exp(-phantom)
        flat = np.ones((2, *phantom.shape[1:]))
        dark = np.zeros((2, *phantom.shape[1:]))
        with h5py.File(str(HDF_FILE), mode='w-') as fp:
            fp.create_dataset('/exchange/data', data=phantom)
            fp.create_dataset('/exchange/data_white', data=flat)
            fp.create_dataset('/exchange/data_dark', data=dark)
    
    def tearDown(self):
        # Remove the temporary HDF5 file
        if os.path.exists(str(HDF_FILE)):
            os.remove(str(HDF_FILE))
        # Remove the reconstructed output
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
    def test_basic_reconstruction(self):
        """Check that a basic reconstruction completes and produces output tiff files."""
        params = make_params()
        params.reconstruction_type = 'slice'
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())
