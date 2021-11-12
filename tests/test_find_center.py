from unittest import TestCase
import argparse
from pathlib import Path


import numpy as np
import h5py
import yaml
import tomopy


from tomopy_cli.find_center import find_rotation_axis


TEST_DIR = Path(__file__).resolve().parent
HDF_FILE = TEST_DIR / 'test_tomogram.h5'
YAML_FILE = TEST_DIR / 'test_tomogram.yaml'


def make_params():
    params = argparse.Namespace()
    params.file_name = HDF_FILE
    params.parameter_file = YAML_FILE
    params.nsino = 0.5
    params.start_proj = 0
    params.end_proj = -1
    params.pixel_size_auto = False
    params.filter_1_material = None
    params.filter_1_auto = False
    params.filter_2_material = None
    params.filter_2_auto = False
    params.filter_3_auto = False
    params.scintillator_auto = False
    params.beam_hardening_method = 'none'
    params.flat_correction_method = 'standard'
    params.binning = 0
    params.file_type = 'standard'
    params.file_format = 'dx'
    params.reverse = False
    params.blocked_views = False
    params.rotation_axis = -1.0
    params.zinger_removal_method = 'none'
    params.dark_zero = False
    params.normalization_cutoff = 1.0
    params.remove_stripe_method = 'none'
    params.retrieve_phase_method = 'none'
    params.minus_log = True
    params.fix_nan_and_inf = False
    params.sinogram_max_value = float('inf')
    return params


class FindRotationAxisTests(TestCase):
    def setUp(self):
        # Remove the temporary HDF5 file
        if HDF_FILE.exists():
            HDF_FILE.unlink()
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
        if HDF_FILE.exists():
            HDF_FILE.unlink()
        if YAML_FILE.exists():
            YAML_FILE.unlink()
    
    def test_single_file(self):
        params = make_params()
        find_rotation_axis(params)
        self.assertEqual(params.rotation_axis, 31.5)
    
    def test_multiple_files(self):
        params = make_params()
        params.file_name = TEST_DIR
        find_rotation_axis(params)
        self.assertEqual(params.rotation_axis, 31.5)
        # Check YAML file
        with open(YAML_FILE, mode='r') as yfp:
            yaml_output = yaml.safe_load(yfp.read())
            self.assertEqual(yaml_output['test_tomogram.h5']['rotation-axis'], 31.5)
