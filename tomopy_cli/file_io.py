import os
import h5py
import dxchange
import dxchange.reader as dxreader
import numpy as np

from tomopy_cli import log

def get_dx_dims(fname, dataset):
    """
    Read array size of a specific group of Data Exchange file.
    Parameters
    ----------
    fname : str
        String defining the path of file or file name.
    dataset : str
        Path to the dataset inside hdf5 file where data is located.
    Returns
    -------
    ndarray
        Data set size.
    """

    grp = '/'.join(['exchange', dataset])

    with h5py.File(fname, "r") as f:
        try:
            data = f[grp]
        except KeyError:
            return None

        shape = data.shape

    return shape

def file_base_name(file_name):
    if '.' in file_name:
        separator_index = file_name.index('.')
        base_name = file_name[:separator_index]
        return base_name
    else:
        return file_name

def path_base_name(path):
    file_name = os.path.basename(path)
    return file_base_name(file_name)


def read_rot_centers(fname):

    try:
        with open(fname) as json_file:
            json_string = json_file.read()
            dictionary = json.loads(json_string)

        return collections.OrderedDict(sorted(dictionary.items()))

    except Exception as error: 
        log.error("ERROR: the json file containing the rotation axis locations is missing")
        log.error("ERROR: run: python find_center.py to create one first")
        exit()

def read_tomo(params, sino):
    # Read APS 32-BM raw data.
    proj, flat, dark, theta = dxchange.read_aps_32id(params.hdf_file, sino=sino)

    if params.hdf_file_type == 'reverse':
        step_size = (theta[1] - theta[0]) 
        theta_size = dxreader.read_dx_dims(params.hdf_file, 'data')[0]
        theta = np.linspace(np.pi , (0+step_size), theta_size)   
        log.warning("  *** overwrite theta")

    if params.hdf_file_type == 'blocked_views':
        miss_angles = [params.missing_angles_start, params.missing_angle_end]
        
        # Manage the missing angles:
        proj = np.concatenate((proj[0:miss_angles[0],:,:], proj[miss_angles[1]+1:-1,:,:]), axis=0)
        theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))

    return proj, flat, dark, theta