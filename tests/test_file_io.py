import unittest
from unittest import mock
import os

import h5py
import numpy as np
from tomopy.misc import phantom
from tomopy import misc, project
import matplotlib.pyplot as plt

from tomopy_cli import file_io

class FlipAndStitchTests(unittest.TestCase):
    def test_tomogram_360(self):
        # Prepare mocked config parameters
        params = mock.MagicMock()
        params.rotation_axis_flip = 10
        flip_axis = 82
        # Prepare simulated flipped data
        original = misc.phantom.shepp2d((128, 128), dtype='float32')
        theta360 = np.linspace(0, np.pi*2, num=362)
        fullsino = project(original, theta=theta360)
        sino360 = fullsino[:,:,flip_axis:]
        # Flip and stitch to simulated data
        sino180, flat180, dark180 = file_io.flip_and_stitch(
            params, img360=sino360, flat360=sino360, dark360=sino360)
        # Test the result
        self.assertEqual(sino180.shape, (181, 1, 183))


class ReadParamTests(unittest.TestCase):
    test_hdf_file = 'filter-tests.hdf5'
    # Sample filters from 7-BM-B for 'open' and 'Cu_1000um' filters
    filter_open = np.array([[79, 112, 101, 110,] + [0,] * 252], dtype='int8')
    filter_Cu_1000um = np.array([[67, 117,  95,  49,  48,  48, 48, 117, 109,] + [0,] * 247], dtype='int8')
    
    def tearDown(self):
        # Clean up the mocked HDF5 file
        if os.path.exists(self.test_hdf_file):
            os.remove(self.test_hdf_file)
    
    def prepare_hdf_file(self, filter_1=None, filter_2=None):
        with h5py.File(self.test_hdf_file, mode='x') as h5fp:
            if filter_1 is not None:
                h5fp['measurement/instrument/filters/Filter_1_Material'] = filter_1
            if filter_2 is not None:
                h5fp['measurement/instrument/filters/Filter_2_Material'] = filter_2
    
    def test_filter_str_to_params(self):
        self.assertEqual(
            file_io._filter_str_to_params('Open'),
            ('Al', 0.),
        )
        self.assertEqual(
            file_io._filter_str_to_params('Cu_1000um'),
            ('Cu', 1000.),
        )
        self.assertEqual(
            file_io._filter_str_to_params('Pb_102.0'),
            ('Pb', 102.),
        )
        self.assertEqual(
            file_io._filter_str_to_params('gibberish'),
            ('Al', 0.),
        )
        self.assertEqual(
            file_io._filter_str_to_params('LuAG_Ce_1000um'),
            ('LuAG_Ce', 1000.),
        )
            
    def test_read_real_filter_materials(self):
        # Prepare mocked HDF5 file
        self.prepare_hdf_file(filter_1=self.filter_open, filter_2=self.filter_Cu_1000um)
        # Prepare mocked config parameters
        params = mock.MagicMock()
        params.file_name = self.test_hdf_file
        params.filter_1_material = 'auto'
        params.filter_1_thickness = 0.
        params.filter_2_material = 'auto'
        params.filter_2_thickness = 0.
        # Call code under test
        file_io.read_filter_materials(params)
        # Check that the params were updated correctly
        self.assertEqual(params.filter_1_material, 'Al')
        self.assertEqual(params.filter_1_thickness, 0.)
        self.assertEqual(params.filter_2_material, 'Cu')
        self.assertEqual(params.filter_2_thickness, 1000.)
    
    def test_read_missing_filter_materials(self):
        # Prepare mocked HDF5 file
        self.prepare_hdf_file(filter_1=None, filter_2=None)
        # Prepare mocked config parameters
        params = mock.MagicMock()
        params.file_name = self.test_hdf_file
        params.filter_1_material = 'auto'
        params.filter_1_thickness = 100.
        params.filter_2_material = 'auto'
        params.filter_2_thickness = 100.
        # Call code under test
        import logging
        log = logging.getLogger(__name__)
        file_io.read_filter_materials(params)
        # Check that the params were updated correctly
        self.assertEqual(params.filter_1_material, 'Al')
        self.assertEqual(params.filter_1_thickness, 0.)
        self.assertEqual(params.filter_2_material, 'Al')
        self.assertEqual(params.filter_2_thickness, 0.)
