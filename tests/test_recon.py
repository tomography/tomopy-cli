from unittest import TestCase, mock
import os
import shutil
from pathlib import Path

import numpy as np
import h5py
import yaml

import tomopy
from tomopy_cli.recon import rec, reconstruction_folder

TEST_DIR = Path(__file__).resolve().parent
HDF_FILE = TEST_DIR / 'test_tomogram.h5'
YAML_FILE = TEST_DIR / 'test_tomogram.yaml'

def make_params():
    params = mock.MagicMock()
    params.file_name = HDF_FILE
    params.save_folder = "{file_name_parent}/_rec"
    params.parameter_file = os.devnull
    params.rotation_axis = 32
    params.file_type = 'standard'
    params.file_format = 'dx'
    params.start_row = 0
    params.end_row = -1
    params.binning = 0
    params.nsino_per_chunk = 256
    params.start_proj = 0
    params.end_proj = -1
    params.flat_correction_method = 'standard'
    params.reconstruction_algorithm = 'gridrec'
    params.gridrec_filter = 'parzen'
    params.reconstruction_mask_ratio = 1.0
    params.reconstruction_type = 'slice'
    params.scintillator_auto = False
    params.blocked_views = False
    params.sinogram_max_value = float('inf')
    return params


class ReconTestBase(TestCase):
    output_dir = Path(__file__).resolve().parent / '_rec'
    output_hdf = Path(__file__).resolve().parent / '_rec' / 'test_tomogram_rec.hdf'
    full_tiff_dir = Path(__file__).resolve().parent / '_rec' / 'test_tomogram_rec'
    
    def setUp(self):
        # Remove the temporary HDF5 file
        if HDF_FILE.exists():
            HDF_FILE.unlink()
        # Prepare some dummy data
        phantom = tomopy.misc.phantom.shepp3d(size=64)
        phantom = np.exp(-phantom)
        theta = np.linspace(0, np.pi, num=64)
        proj = tomopy.sim.project.project(phantom, theta, pad=False)
        flat = np.ones((2, *phantom.shape[1:]))
        dark = np.zeros((2, *phantom.shape[1:]))
        with h5py.File(HDF_FILE, mode='w-') as fp:
            fp.create_dataset('/exchange/data', data=proj)
            fp.create_dataset('/exchange/data_white', data=flat)
            fp.create_dataset('/exchange/data_dark', data=dark)
            fp.create_dataset('/exchange/theta', data=theta)
    
    def tearDown(self):
        # Remove the temporary HDF5 file
        if HDF_FILE.exists():
            HDF_FILE.unlink()
        # Remove the reconstructed output
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)


class ReconTests(ReconTestBase):
    def test_slice_reconstruction(self):
        """Check that a basic reconstruction completes and produces output tiff files."""
        params = make_params()
        params.reconstruction_type = 'slice'
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())
    
    def test_full_reconstruction(self):
        """Check that a basic reconstruction completes and produces output tiff files."""
        params = make_params()
        params.reconstruction_type = 'full'
        params.save_format = 'tiff'
        response = rec(params=params)
        # import pdb; pdb.set_trace()
        self.assertTrue(self.full_tiff_dir.exists())
    
    def test_hdf_output(self):
        params = make_params()
        params.reconstruction_type = 'full'
        params.save_format = "h5"
        response = rec(params=params)
        expected_hdf5path = self.output_hdf
        # Check that tiffs are not saved and HDF5 file is saved
        self.assertFalse(os.path.exists(self.full_tiff_dir))
        self.assertTrue(os.path.exists(expected_hdf5path))
        with h5py.File(expected_hdf5path, mode='r') as h5fp:
            vol = h5fp['volume']
            self.assertEqual(vol.shape, (64, 64, 64))
            self.assertFalse(np.any(np.isnan(vol)))
    
    def test_hdf_output_chunks(self):
        # Test with multiple chunks to ensure they're all written
        params = make_params()
        params.reconstruction_type = 'full'
        params.save_format = 'h5'
        params.nsino_per_chunk = 16 # 4 chunks
        response = rec(params=params)
        expected_hdf5path = self.output_hdf
        # Check that tiffs are not saved and HDF5 file is saved
        self.assertFalse(os.path.exists(self.full_tiff_dir))
        self.assertTrue(os.path.exists(expected_hdf5path))
        with h5py.File(expected_hdf5path, mode='r') as h5fp:
            vol = h5fp['volume']
            self.assertEqual(vol.shape, (64, 64, 64))
            self.assertFalse(np.any(np.isnan(vol)))

    def test_reconstruction_folder(self):
        params = make_params()
        params.save_folder = "_rec"
        output = reconstruction_folder(params)
        self.assertEqual(str(output), "_rec")
        # Check config parameters
        params = make_params()
        params.save_folder = "_rec_{reconstruction_algorithm}/"
        params.reconstruction_algorithm = "sirt"
        output = reconstruction_folder(params)
        self.assertEqual(str(output), "_rec_sirt")
        # Check parent file name for a file
        this_file = Path(__file__).resolve()
        params = make_params()
        params.file_name = str(this_file)
        params.save_folder = "{file_name_parent}_rec/"
        output = reconstruction_folder(params)
        self.assertEqual(str(output), str(this_file.parent) + "_rec")
        # Check parent file name for a directory
        this_file = Path(__file__).resolve()
        params = make_params()
        params.file_name = str(this_file.parent) + '/'
        params.save_folder = "{file_name_parent}_rec/"
        output = reconstruction_folder(params)
        self.assertEqual(str(output), str(this_file.parent) + "_rec")


class YamlParamsTests(ReconTestBase):
    yaml_file = TEST_DIR / "my_files.yaml"
    def setUp(self):
        # Delete any old files 
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
        # Create a new YAML file
        opts = {
            "test_tomogram.h5": {
                "spam": "foo",
                "rotation-axis": 1200,
            }
        }
        with open(self.yaml_file, mode='w') as fp:
            fp.write(yaml.dump(opts))
        super().setUp()
    
    def tearDown(self):
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
        super().tearDown()

    def test_yaml_params(self):
        params = make_params()
        params.parameter_file = self.yaml_file
        response = rec(params=params)


class TryCenterTests(ReconTestBase):
    yaml_file = TEST_DIR / "my_files.yaml"
    
    def setUp(self):
        # Delete any old files 
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
        super().setUp()
    
    def tearDown(self):
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
        super().tearDown()

    def test_recon_output_dir(self):
        """Check that ``--reconstruction-type=try`` respects output
        directory.
        
        """
        params = make_params()
        params.reconstruction_type = "try"
        params.center_search_width = 10
        params.save_folder = "{file_name_parent}/_rec"
        params.parameter_file = os.devnull
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())

    def test_extra_args_no_file(self):
        """Check for behavior if the file is or isn't present in the
        extra_args yaml file.
        
        """
        params = make_params()
        params.reconstruction_type = "try"
        params.center_search_width = 10
        params.save_folder = "{file_name_parent}/_rec"
        params.parameter_file = self.yaml_file
        # Create the YAML file
        opts = {
            "other_tomogram.h5": {
                "spam": "foo",
                "rotation-axis": 1200,
            }
        }
        with open(self.yaml_file, mode='w') as fp:
            fp.write(yaml.dump(opts))
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())
    
    def test_extra_args_no_rotation_axis(self):
        """Check for behavior if the file is or isn't present in the
        extra_args yaml file.
        
        """
        params = make_params()
        params.reconstruction_type = "try"
        params.center_search_width = 10
        params.save_folder = "{file_name_parent}/_rec"
        params.parameter_file = self.yaml_file
        # Create the YAML file
        opts = {
            "test_tomogram.h5": {
                "spam": "foo",
            }
        }
        with open(self.yaml_file, mode='w') as fp:
            fp.write(yaml.dump(opts))
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())

    def test_extra_args_no_file(self):
        """Check for behavior if the file is or isn't present in the
        extra_args yaml file.
        
        """
        params = make_params()
        params.reconstruction_type = "try"
        params.center_search_width = 10
        params.save_folder = "{file_name_parent}/_rec"
        params.parameter_file = "/tmp/gweoiuwerw"
        response = rec(params=params)
        self.assertTrue(self.output_dir.exists())
