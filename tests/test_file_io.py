import unittest
from unittest import mock

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
