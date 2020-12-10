import unittest
from unittest import mock
from pathlib import Path
import sys

# Import the tomopy executable file
from importlib.util import spec_from_loader, module_from_spec
from importlib.machinery import SourceFileLoader 
bin_dir = str(Path(__file__).parent.parent / "bin" / "tomopy")
spec = spec_from_loader("tomopy", SourceFileLoader("tomopy", bin_dir))
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


class RotationAxisFileTests(unittest.TestCase):
    """Check that file names can be read from a rotation-axis.json
    file.
    
    """
    rot_axis_file = Path(__file__).resolve().parent / 'rotation_axis.json'
    def setUp(self):
        if self.rot_axis_file.exists():
            self.rot_axis_file.unlink()
        # Create a rotation_axis.json file
        with open(self.rot_axis_file, mode='x') as fp:
            fp.write('{"0": {"test_tomogram.h5": 1287.25}}')
    
    def tearDown(self):
        if self.rot_axis_file.exists():
            self.rot_axis_file.unlink()

    @mock.patch('tomopy_bin.recon.rec')
    @mock.patch('tomopy_bin.config.update_config')
    def test_filename_from_json_file(self, update_config_func, rec_func):
        params = make_params()
        params.file_name = self.rot_axis_file
        response = tomopy_bin.run_rec(params)
        #self.assertEqual(str(params.file_name), '/home/mwolf/src/tomopy-cli/tests/test_tomogram.h5')
        self.assertEqual(params.rotation_axis, 32)

