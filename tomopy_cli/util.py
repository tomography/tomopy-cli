import logging
from collections.abc import Mapping

import numpy as np


log = logging.getLogger(__name__)


def theta_step(start, end, proj_number):
    return (end-start)/proj_number

def positive_int(value):
    """Convert *value* to an integer and make sure it is positive."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Only positive integers are allowed')

    return result

def range_list(value):
    """
    Split *value* separated by ':' into int triple, filling missing values with 1s.
    """
    def check(region):
        if region[0] >= region[1]:
            raise argparse.ArgumentTypeError("{} must be less than {}".format(region[0], region[1]))

    lst = [int(x) for x in value.split(':')]

    if len(lst) == 1:
        frm = lst[0]
        return (frm, frm + 1, 1)

    if len(lst) == 2:
        check(lst)
        return (lst[0], lst[1], 1)

    if len(lst) == 3:
        check(lst)
        return (lst[0], lst[1], lst[2])

    raise argparse.ArgumentTypeError("Cannot parse {}".format(value))

def restricted_float(x):

    x = float(x)
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x
    

def guess_center(first_projection, last_projection):
    """
    Compute the tomographic rotation center based on cross-correlation technique.
    *first_projection* is the projection at 0 deg, *last_projection* is the
    projection at 180 deg.
    """
    from scipy.signal import fftconvolve
    width = first_projection.shape[1]
    first_projection = first_projection - first_projection.mean()
    last_projection = last_projection - last_projection.mean()

    # The rotation by 180 deg flips the image horizontally, in order
    # to do cross-correlation by convolution we must also flip it
    # vertically, so the image is transposed and we can apply convolution
    # which will act as cross-correlation
    convolved = fftconvolve(first_projection, last_projection[::-1, :], mode='same')
    center = np.unravel_index(convolved.argmax(), convolved.shape)[1]

    return (width / 2.0 + center) / 2


class CenterCalibration(object):

    def __init__(self, first, last):
        self.center = guess_center(first, last)
        self.height, self.width = first.shape

    @property
    def position(self):
        return self.width / 2.0 + self.width - self.center * 2.0

    @position.setter
    def position(self, p):
        self.center = (self.width / 2.0 + self.width - p) / 2


def update_dict(original: Mapping, new: Mapping)->Mapping:
    """Recursively update a dictionary in place with new values.

    This is distinct from the python ``dict.update`` method in that it
    respects existing entries and just updates them with new values.

    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    Parameters
    ==========
    original
      The target dictionary that will be updated.
    new
      The dictionary with new values that will be added to *original*.

    Returns
    =======
    original
      Same dictionary that was passed, with values modified in place.

    """
    for k, v in new.items():
        if isinstance(v, Mapping):
            original[k] = update_dict(original.get(k, {}), v)
        else:
            original[k] = v
    return original
