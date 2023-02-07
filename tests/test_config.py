import argparse
from unittest import TestCase, mock
from pathlib import Path
import os
import shutil

import yaml

from tomopy_cli import config

TEST_DIR = Path(__file__).resolve().parent
HDF_FILE = TEST_DIR / 'test_tomogram.h5'


def make_params():
    params = mock.MagicMock()
    params.config_update = False
    params.dx_update = False
    params.reconstruction_type = "full"
    params.file_name = HDF_FILE
    params.config = Path("test_recon.conf")
    params.save_folder = "{file_name_parent}/_rec"
    return params


class TestUpdateConfig(TestCase):
    def test_full_recon_config_hdf(self):
        """Test that the config file is saved for an HDF file."""
        params = make_params()
        params.save_format = "h5"
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
        params.save_format = "tiff"
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


class NamespaceTests(TestCase):
    def test_config(self):
        pass


class YamlParamsTests(TestCase):
    yaml_file = TEST_DIR / "my_files.yaml"
    def setUp(self):
        # Delete any old files 
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
        # Create a new YAML file
        opts = {
            "my_tomo_file.h5": {
                "spam": "foo",
                "rotation-axis": 1200,
            }
        }
        with open(self.yaml_file, mode='w') as fp:
            fp.write(yaml.dump(opts))
    
    def tearDown(self):
        if self.yaml_file.exists():
            os.remove(self.yaml_file)
    
    def test_yaml_args(self):
        args = argparse.Namespace(spam="eggs")
        new_args = config.yaml_args(args, yaml_file=self.yaml_file, sample="my_tomo_file.h5")
        # Check that new args were set
        self.assertIsNot(new_args, args, msg="``yaml_args`` should return a deep copy")
        self.assertEqual(new_args.spam, "foo")
        self.assertEqual(new_args.file_name, Path("my_tomo_file.h5"))
        # Check that original args are unchanged
        self.assertEqual(args.spam, "eggs")
    
    def test_yaml_with_cli_args(self):
        # Test with ``--spam=eggs`` style CLI arg
        args = argparse.Namespace(spam="eggs")
        cli_args = ['tomopy', 'recon', '--spam=eggs']
        new_args = config.yaml_args(args, yaml_file=self.yaml_file, sample="my_tomo_file.h5", cli_args=cli_args)
        # Check that new args were set
        self.assertEqual(new_args.spam, "eggs")
        self.assertEqual(new_args.file_name, Path("my_tomo_file.h5"))
        # Test with ``--spam eggs`` style CLI arg
        args = argparse.Namespace(spam="eggs")
        cli_args = ['tomopy', 'recon', '--spam=eggs']
        new_args = config.yaml_args(args, yaml_file=self.yaml_file, sample="my_tomo_file.h5", cli_args=cli_args)
        # Check that new args were set
        self.assertEqual(new_args.spam, "eggs")
        self.assertEqual(new_args.file_name, Path("my_tomo_file.h5"))
        # Test with a similar but slightly different CLI arg
        args = argparse.Namespace(spam="eggs")
        cli_args = ['tomopy', 'recon', '--spammery=eggs']
        new_args = config.yaml_args(args, yaml_file=self.yaml_file, sample="my_tomo_file.h5", cli_args=cli_args)
        # Check that new args were set
        self.assertEqual(new_args.spam, "foo")
        self.assertEqual(new_args.file_name, Path("my_tomo_file.h5"))
