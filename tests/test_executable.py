import unittest
from unittest import mock
from pathlib import Path
import sys

# Import the tomopy executable file
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader 
bin_dir = str(Path(__file__).parent.parent / "bin" / "tomopycli.py")
spec = spec_from_loader("tomopycli", SourceFileLoader("tomopycli", bin_dir))
tomopy_bin = module_from_spec(spec)
spec.loader.exec_module(tomopy_bin)
sys.modules['tomopy_bin'] = tomopy_bin


def make_params():
    params = mock.MagicMock()
    params.file_name = "test_tomogram.h5"
    params.rotation_axis = 32
    params.file_type = 'standard'
    params.file_format = 'dx'
    params.start_row = 0
    params.end_row = -1
    params.binning = 0
    params.nsino_per_chunk = 256
    params.flat_correction_method = 'standard'
    params.reconstruction_algorithm = 'gridrec'
    params.gridrec_filter = 'parzen'
    params.reconstruction_mask_ratio = 1.0
    params.reconstruction_type = 'slice'
    return params


class YAMLFileTests(unittest.TestCase):
    """Check that file names can be read from a extra_params.yaml
    file.
    
    """
    params_file = Path(__file__).resolve().parent / 'extra_params.yaml'
    def setUp(self):
        if self.params_file.exists():
            self.params_file.unlink()
        # Create a extra_params.yaml file
        with open(self.params_file, mode='x') as fp:
            fp.write('test_tomogram.h5:\n  rotation_axis: 1287.25')
    
    def tearDown(self):
        if self.params_file.exists():
            self.params_file.unlink()
    
    @mock.patch('tomopy_bin.recon.rec')
    @mock.patch('tomopy_bin.config.update_config')
    def test_filename_from_yaml_file(self, update_config_func, rec_func):
        """Doesn't test results, just that the binary calls the rec function."""
        params = make_params()
        params.file_name = self.params_file
        response = tomopy_bin.run_rec(params)
        # Test that only the files are used, but no parameters are overridden
        self.assertEqual(params.rotation_axis, 32, msg="Rotation axis should not be overridden")
        # Test the call to reconstruction function
        rec_func.assert_called_once()
        rec_args = rec_func.call_args_list[0][0][0]
        self.assertEqual(rec_args.file_name, self.params_file.parent / 'test_tomogram.h5')
