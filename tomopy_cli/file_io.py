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

    if params.hdf_file_type == 'flip_and_stich':
        log.info("   *** loading a 360 deg flipped data set")
        proj360, flat360, dark360, theta360 = dxchange.read_aps_32id(params.hdf_file, sino=sino)
        proj, flat, dark = util.flip_and_stitch(variableDict, proj360, flat360, dark360)
        theta = theta360[:len(theta360)//2] # take first half
    else:
        # Read APS 32-BM raw data.
        log.info("  *** loading a stardard data set")
        proj, flat, dark, theta = dxchange.read_aps_32id(params.hdf_file, sino=sino)


    if params.reverse:
        log.info("  *** correcting for 180-0 data collection")
        step_size = (theta[1] - theta[0]) 
        theta_size = dxreader.read_dx_dims(params.hdf_file, 'data')[0]
        theta = np.linspace(np.pi , (0+step_size), theta_size)   

    if params.blocked_views:
        log.info("  *** correcting for blocked view data collection")
        miss_angles = [params.missing_angles_start, params.missing_angle_end]
        
        # Manage the missing angles:
        proj = np.concatenate((proj[0:miss_angles[0],:,:], proj[miss_angles[1]+1:-1,:,:]), axis=0)
        theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))
 
    # new missing projection handling
    # if params.blocked_views:
    #     log.warning("  *** new missing angle handling")
    #     miss_angles = [params.missing_angles_start, params.missing_angle_end]
    #     data = util.patch_projection(data, miss_angles)


    return proj, flat, dark, theta