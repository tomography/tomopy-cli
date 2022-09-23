import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tomopy_cli import beamhardening as bh


TEST_DIR = Path(__file__).resolve().parent
HDF_FILE = TEST_DIR / "meta_mock.h5"


def make_params():
    params = mock.MagicMock()
    params.beam_hardening_method = "standard"
    params.sample_material = "Fe"
    params.scintillator_material = "LuAG_Ce"
    params.scintillator_thickness = 100.0
    params.filter_1_material = "none"
    params.filter_2_material = "none"
    params.filter_3_material = "none"
    params.file_name = HDF_FILE
    # params.file_name = HDF_FILE
    # params.save_folder = "{file_name_parent}/_rec"
    # params.parameter_file = os.devnull
    # params.rotation_axis = 32
    # params.file_type = 'standard'
    # params.file_format = 'dx'
    # params.start_row = 0
    # params.end_row = -1
    # params.binning = 0
    # params.nsino_per_chunk = 256
    # params.flat_correction_method = 'standard'
    # params.reconstruction_algorithm = 'gridrec'
    # params.gridrec_filter = 'parzen'
    # params.reconstruction_mask_ratio = 1.0
    # params.reconstruction_type = 'slice'
    # params.scintillator_auto = False
    # params.blocked_views = False
    return params


class AbsorptionCalcTests(unittest.TestCase):
    def setUp(self):
        self.fake_spectrum = bh.Spectrum(np.array([7.999, 8.000, 8.001]), 
                                        np.array([0.0, 1.0, 0.0]))
        self.Al_filter = bh.Material('Al', 2.7)
        self.Al_filter2 = bh.Material('Al', 2.7)
        self.softener = bh.BeamSoftener(make_params())
        self.softener.scintillator_material = bh.Material('LuAG_Ce', 6.73)
        self.softener.sample_material = self.Al_filter
        self.softener.scintillator_thickness = 100
        self.softener.threshold_trans = 1e-5
 
    def test_mean_energy(self):
        self.assertAlmostEqual(self.fake_spectrum.fmean_energy(), 8.000)
    
    def test_Al_absorption(self):
        self.assertAlmostEqual(self.Al_filter.finterpolate_absorption(8.000), 49.5046)
    
    def test_proj_density(self):
        self.assertAlmostEqual(self.Al_filter.fcompute_proj_density(1.0), 2.7e-4)
        self.assertAlmostEqual(self.Al_filter.fcompute_proj_density(1000.0), 0.27)
    
    def test_Al_transmitted_spectrum(self):
        thickness = 10 #microns
        trans_spec = self.Al_filter.fcompute_transmitted_spectrum(thickness,self.fake_spectrum)
        proj_den = self.Al_filter.fcompute_proj_density(10)
        np.testing.assert_allclose(trans_spec.spectral_power, np.array([0, np.exp(-49.5939 * proj_den), 0]))
    
    def test_Al_absorbed_spectrum(self):
        thickness = 10 #microns
        trans_spec = self.Al_filter.fcompute_absorbed_spectrum(thickness,self.fake_spectrum)
        proj_den = self.Al_filter.fcompute_proj_density(10)
        np.testing.assert_allclose(trans_spec.spectral_power, np.array([0, 1.0 - np.exp(-49.5046 * proj_den), 0]))
    
    def test_fapply_filters(self):
        filters = {self.Al_filter2: 10, self.Al_filter: 30}
        proj_den = self.Al_filter.fcompute_proj_density(40)
        trans_spec = bh.fapply_filters(filters, self.fake_spectrum)
        np.testing.assert_allclose(trans_spec.spectral_power, np.array([0, np.exp(-49.5939 * proj_den), 0]))
        
    def test_ffind_calibration_one_angle(self):
        params = make_params()
        test_spline = self.softener.ffind_calibration_one_angle(self.fake_spectrum)
        # Test that a transmission of 1 gives a pathlength of zero
        self.assertAlmostEqual(test_spline(1), 0)
        # Test to make sure that fixing the values past the threshold value works
        self.assertAlmostEqual(test_spline(1e-8), test_spline(8e-6))
        self.assertAlmostEqual(test_spline(1.3), test_spline(3))
        # Test at a fixed filter thickness of 10, 100, and 1000 microns Al
        sample_trans = np.exp(-49.5046 * 0.0027)
        self.assertAlmostEqual((test_spline(sample_trans)-10)/10, 0, 2)
        sample_trans = np.exp(-49.5046 * 0.027)
        self.assertAlmostEqual((test_spline(sample_trans)-100)/100, 0, 2)
        sample_trans = np.exp(-49.5046 * 0.135)
        self.assertAlmostEqual((test_spline(sample_trans)-500) / 500, 0, 2)


class BeamSoftenerTests(unittest.TestCase):
    def test_init(self):
        params = make_params()
        softener = bh.BeamSoftener(params)


if __name__ == '__main__':
    unittest.main()
