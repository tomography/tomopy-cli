from unittest import TestCase, mock
from pathlib import Path
import os
import shutil

from tomopy_cli import config


HDF_FILE = Path(__file__).resolve().parent / 'test_tomogram.h5'


def make_params():
    params = mock.MagicMock()
    params.config_update = False
    params.dx_update = False
    params.reconstruction_type = "full"
    params.file_name = HDF_FILE
    params.config = Path("test_recon.conf")
    params.output_folder = "{file_name_parent}/_rec"
    return params


class TestUpdateConfig(TestCase):
    def test_full_recon_config_hdf(self):
        """Test that the config file is saved for an HDF file."""
        params = make_params()
        params.output_format = "hdf5"
        output_dir = HDF_FILE.parent / "_rec"
        os.makedirs(output_dir)
        expected_config = output_dir / "test_tomogram_rec_test_recon.conf"
        try:
            config.update_config(params)
            self.assertTrue(expected_config.exists(), expected_config)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)
    
    def test_full_recon_config_tiff_stack(self):
        """Test that the config file is saved for a stack of TIFFs."""
        params = make_params()
        params.output_format = "tiff_stack"
        base_dir = HDF_FILE.parent / "_rec"
        os.makedirs(base_dir)
        output_dir = base_dir / "test_tomogram_rec"
        os.makedirs(output_dir)
        expected_config = output_dir / "test_recon.conf"
        try:
            config.update_config(params)
            self.assertTrue(expected_config.exists(), expected_config)
        finally:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            if base_dir.exists():
                shutil.rmtree(base_dir)
