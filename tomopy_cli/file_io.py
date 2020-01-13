import os
import h5py
import json
import collections
import tomopy
import dxchange
import dxchange.reader as dxreader
import numpy as np

from tomopy_cli import log
from tomopy_cli import proc

 
def read_tomo(sino, params):
    """
    Read in the tomography data.
    Inputs:
    sino: tuple of (start_row, end_row) to be read in
    params: parameters for reconstruction
    Output:
    projection data
    flat field (bright) data
    dark field data
    theta: Numpy array of angle for each projection
    rotation_axis: location of the rotation axis
    """
    if params.hdf_file_type == 'standard':
        # Read APS 32-BM raw data.
        log.info("  *** loading a stardard data set: %s" % params.hdf_file)
        proj, flat, dark, theta = dxchange.read_aps_32id(params.hdf_file, sino=sino)
    elif params.hdf_file_type == 'flip_and_stich':
        log.info("   *** loading a 360 deg flipped data set: %s" % params.hdf_file)
        proj360, flat360, dark360, theta360 = dxchange.read_aps_32id(params.hdf_file, sino=sino)
        proj, flat, dark = flip_and_stitch(variableDict, proj360, flat360, dark360)
        theta = theta360[:len(theta360)//2] # take first half
    else: # params.hdf_file_type == 'mosaic':
        log.error("   *** loading a mosaic data set is not supported yet")
        exit()

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
    #     data = patch_projection(data, miss_angles)


    rotation_axis = params.rotation_axis / np.power(2, float(params.binning))
    if (params.binning == 0):
        log.info("  *** rotation center: %f" % rotation_axis)
    else:
        log.warning("  *** binning: %d" % params.binning)
        log.warning("  *** rotation center: %f" % rotation_axis)


    proj = binning(proj, params)
    flat = binning(flat, params)
    dark = binning(dark, params)

    return proj, flat, dark, theta, rotation_axis


def _binning(data, params):

    data = tomopy.downsample(data, level=int(params.binning), axis=2) 
    data = tomopy.downsample(data, level=int(params.binning), axis=1)

    return data


def flip_and_stitch(params, img360, flat360, dark360):

    center = int(params.rotation_axis_flip)
    img = np.zeros([img360.shape[0]//2,img360.shape[1],2*img360.shape[2]-2*center],dtype=img360.dtype)
    flat = np.zeros([flat360.shape[0],flat360.shape[1],2*flat360.shape[2]-2*center],dtype=img360.dtype)
    dark = np.zeros([dark360.shape[0],dark360.shape[1],2*dark360.shape[2]-2*center],dtype=img360.dtype)
    img[:,:,img360.shape[2]-2*center:] = img360[:img360.shape[0]//2,:,:]
    img[:,:,:img360.shape[2]] = img360[img360.shape[0]//2:,:,::-1]
    flat[:,:,flat360.shape[2]-2*center:] = flat360
    flat[:,:,:flat360.shape[2]] = flat360[:,:,::-1]
    dark[:,:,dark360.shape[2]-2*center:] = dark360
    dark[:,:,:dark360.shape[2]] = dark360[:,:,::-1]

    params.rotation_axis = img.shape[2]//2

    return img, flat, dark


def patch_projection(data, miss_angles):

    fdatanew = np.fft.fft(data,axis=2)

    w = int((miss_angles[1]-miss_angles[0]) * 0.3)

    fdatanew[miss_angles[0]:miss_angles[0]+w,:,:] = np.fft.fft(data[miss_angles[0]-1,:,:],axis=1)
    fdatanew[miss_angles[0]:miss_angles[0]+w,:,:] *= np.reshape(np.cos(np.pi/2*np.linspace(0,1,w)),[w,1,1])

    fdatanew[miss_angles[1]-w:miss_angles[1],:,:] = np.fft.fft(data[miss_angles[1]+1,:,:],axis=1)
    fdatanew[miss_angles[1]-w:miss_angles[1],:,:] *= np.reshape(np.sin(np.pi/2*np.linspace(0,1,w)),[w,1,1])

    fdatanew[miss_angles[0]+w:miss_angles[1]-w,:,:] = 0
    # lib.warning("  *** %d, %d, %d " % (datanew.shape[0], datanew.shape[1], datanew.shape[2]))

    lib.warning("  *** patch_projection")
    slider(np.log(np.abs(fdatanew.swapaxes(0,1))), axis=0)
    a = np.real(np.fft.ifft(fdatanew,axis=2))
    b = np.imag(np.fft.ifft(fdatanew,axis=2))
    # print(a.shape)
    slider(a.swapaxes(0,1), axis=0)
    slider(b.swapaxes(0,1), axis=0)
    return np.real(np.fft.ifft(fdatanew,axis=2))


def get_dx_dims(params):
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

    dataset='data'

    grp = '/'.join(['exchange', dataset])

    with h5py.File(params.hdf_file, "r") as f:
        try:
            data = f[grp]
        except KeyError:
            return None

        shape = data.shape

    return shape


def file_base_name(fname):
    if '.' in fname:
        separator_index = fname.index('.')
        base_name = fname[:separator_index]
        return base_name
    else:
        return fname

def path_base_name(path):
    fname = os.path.basename(path)
    return file_base_name(fname)


def read_rot_centers(params):

    # Add a trailing slash if missing
    top = os.path.join(params.hdf_file, '')

    # Load the the rotation axis positions.
    jfname = top + "rotation_axis.json"
    
    try:
        with open(jfname) as json_file:
            json_string = json_file.read()
            dictionary = json.loads(json_string)

        return collections.OrderedDict(sorted(dictionary.items()))

    except Exception as error: 
        log.error("ERROR: the json %s file containing the rotation axis locations is missing" % jfname)
        log.error("ERROR: to create one run:")
        log.error("ERROR: $ tomopy find_center --hdf-file %s" % top)
        exit()

def read_rot_center(params):
    """
    Read the rotation center from /process group in the DXchange file.
    Return: rotation center from this dataset or None if it doesn't exist.
    """
    with h5py.File(params.hdf_file) as hdf_file:
        try:
            rot_center = hdf_file['/process/rot_center'][0]
            log.info('Rotation center read from HDF5 file: {0:f}'.format(rot_center)) 
            return rot_center
        except KeyError:
            log.warning('No rotation center stored in the HDF5 file.')
            return None
