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


class WriteHDF5Tests(unittest.TestCase):
    hdf_filename = 'test_output.h5'
    
    def tearDown(self):
        if os.path.exists(self.hdf_filename):
            os.remove(self.hdf_filename)
    
    def test_write_hdf5_full(self):
        data = phantom.shepp3d(size=32, dtype='float16')
        file_io.write_hdf5(data, fname=self.hdf_filename)
        # Check that the data were written properly
        with h5py.File(self.hdf_filename, mode='r') as h5fp:
            self.assertIn('volume', h5fp)
            np.testing.assert_array_equal(h5fp['volume'], data)
    
    def test_write_hdf5_partial(self):
        data = phantom.shepp3d(size=32, dtype='float16')
        data_partial = data[10:16]
        # Write the data to the HDF5 file
        file_io.write_hdf5(data_partial, fname=self.hdf_filename, maxsize=data.shape, dest_idx=slice(10,16))
        # Check that the data were written properly
        with h5py.File(self.hdf_filename, mode='r') as h5fp:
            self.assertIn('volume', h5fp)
            # Check that the target data were saved in the right place
            np.testing.assert_array_equal(h5fp['volume'][10:16], data_partial)
            # Check that the rest of the data are zero
            self.assertTrue(np.all(np.isnan(h5fp['volume'][:10])))
            self.assertTrue(np.all(np.isnan(h5fp['volume'][16:])))
    
    def test_write_hdf5_full_overwrite(self):
        # Create some existing data to overwrite
        with h5py.File(self.hdf_filename, mode='a') as h5fp:
            h5fp.create_dataset('volume', data=np.empty((8, 8, 8)))
        # Create a dummy dataset to reconstruct
        data = phantom.shepp3d(size=32, dtype='float16')
        data_partial = data[10:16]
        # Write the data to the HDF5 file
        file_io.write_hdf5(data_partial, fname=self.hdf_filename,
                           maxsize=data.shape, dest_idx=slice(10,16),
                           overwrite=True)
        # Check that the data were written properly
        with h5py.File(self.hdf_filename, mode='r') as h5fp:
            self.assertIn('volume', h5fp)
            # Check that the target data were saved in the right place
            np.testing.assert_array_equal(h5fp['volume'][10:16], data_partial)
            # Check that the rest of the data are zero
            self.assertTrue(np.all(np.isnan(h5fp['volume'][:10])))
            self.assertTrue(np.all(np.isnan(h5fp['volume'][16:])))
    
    def test_write_hdf5_partial_overwrite(self):
        # Create some existing data to overwrite
        with h5py.File(self.hdf_filename, mode='a') as h5fp:
            h5fp.create_dataset('volume', data=np.ones((8, 8, 8)))
        data = phantom.shepp3d(size=32, dtype='float32')
        data_partial = data[10:16]
        data_partial_a = data[10:13]
        data_partial_b = data[13:16]
        # Write the data to the HDF5 file
        file_io.write_hdf5(data_partial_a, fname=self.hdf_filename,
                           maxsize=data.shape, dest_idx=slice(10,13), overwrite=True)
        file_io.write_hdf5(data_partial_b, fname=self.hdf_filename,
                           maxsize=data.shape, dest_idx=slice(13,16), overwrite=False)
        # Check that the data were written properly
        with h5py.File(self.hdf_filename, mode='r') as h5fp:
            self.assertIn('volume', h5fp)
            # Check that the target data were saved in the right place
            np.testing.assert_array_equal(h5fp['volume'][10:16], data_partial)
            # Check that the rest of the data are np.nan
            self.assertTrue(np.all(np.isnan(h5fp['volume'][:10])))
            self.assertTrue(np.all(np.isnan(h5fp['volume'][16:])))
    
    def test_write_hdf5_mismatched_dtypes(self):
        """Try to write partial data with mismatch dtypes should fail."""
        # Create some existing data to overwrite
        with h5py.File(self.hdf_filename, mode='a') as h5fp:
            h5fp.create_dataset('volume', data=np.ones((8, 8, 8)))
        data = phantom.shepp3d(size=8, dtype='float32')
        # Write the data to the HDF5 file
        with self.assertRaises(TypeError):
            file_io.write_hdf5(data[10:13], fname=self.hdf_filename,
                               maxsize=data.shape, dest_idx=slice(10,13),
                               overwrite=False, dtype='int')
