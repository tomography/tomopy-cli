import os
import h5py
import json
import collections
import tomopy
import dxchange
import dxchange.reader as dxreader
import numpy as np

from tomopy_cli import log
from tomopy_cli import __version__
from tomopy_cli import find_center
from tomopy_cli import config
from tomopy_cli import beamhardening

 
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
    if params.file_type == 'standard':
        # Read APS 32-BM raw data.
        log.info("  *** loading a stardard data set: %s" % params.file_name)
        proj, flat, dark, theta = _read_tomo(params, sino=sino)
    elif params.file_type == 'flip_and_stich':
        log.info("   *** loading a 360 deg flipped data set: %s" % params.file_name)
        proj360, flat360, dark360, theta360 = _read_tomo(params, sino=sino)
        proj, flat, dark = flip_and_stitch(variableDict, proj360, flat360, dark360)
        theta = theta360[:len(theta360)//2] # take first half
    else: # params.file_type == 'mosaic':
        log.error("   *** loading a mosaic data set is not supported yet")
        exit()

    if params.reverse:
        log.info("  *** correcting for 180-0 data collection")
        step_size = (theta[1] - theta[0]) 
        theta_size = _read_theta_size(params)
        theta = np.linspace(np.pi , (0+step_size), theta_size)   

    proj, theta = blocked_view(proj, theta, params)

    # new missing projection handling
    # if params.blocked_views:
    #     log.warning("  *** new missing angle handling")
    #     miss_angles = [params.missing_angles_start, params.missing_angle_end]
    #     data = patch_projection(data, miss_angles)

    proj, flat, dark = binning(proj, flat, dark, params)

    rotation_axis = params.rotation_axis / np.power(2, float(params.binning))
    log.info("  *** rotation center: %f" % rotation_axis)

    return proj, flat, dark, theta, rotation_axis


def _read_theta_size(params):
    if (str(params.file_type) in {'dx', 'aps2bm', 'aps7bm', 'aps32id'}):
        theta_size = dxreader.read_dx_dims(params.file_name, 'data')[0]
    # elif:
    #     # add here other reader of theta size for other formats
    #     log.info("  *** %s is a valid xxx file format" % params.file_name)
    else:
        log.error("  *** %s is not a supported file format" % params.file_format)
        exit()

    return theta_size


def _read_tomo(params, sino):

    if (str(params.file_format) in {'dx', 'aps2bm', 'aps7bm', 'aps32id'}):
        proj, flat, dark, theta = dxchange.read_aps_32id(params.file_name, sino=sino)
        log.info("  *** %s is a valid dx file format" % params.file_name)
    # elif:
    #     # add here other dxchange loader
    #     log.info("  *** %s is a valid xxx file format" % params.file_name)

    else:
        log.error("  *** %s is not a supported file format" % params.file_format)
        exit()
    return proj, flat, dark, theta


def blocked_view(proj, theta, params):
    log.info("  *** correcting for blocked view data collection")
    if params.blocked_views:
        log.warning('  *** *** ON')
        miss_angles = [params.missing_angles_start, params.missing_angles_end]
        
        # Manage the missing angles:
        proj = np.concatenate((proj[0:miss_angles[0],:,:], proj[miss_angles[1]+1:-1,:,:]), axis=0)
        theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))
    else:
        log.warning('  *** *** OFF')

    return proj, theta


def binning(proj, flat, dark, params):

    log.info("  *** binning")
    if(params.binning == 0):
        log.info('  *** *** OFF')
    else:
        log.warning('  *** *** ON')
        log.warning('  *** *** binning: %d' % params.binning)
        proj = _binning(proj, params)
        flat = _binning(flat, params)
        dark = _binning(dark, params)

    return proj, flat, dark

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

    with h5py.File(params.file_name, "r") as f:
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
    top = os.path.join(params.file_name, '')

    # Load the the rotation axis positions.
    jfname = top + params.rotation_axis_file
    
    try:
        with open(jfname) as json_file:
            json_string = json_file.read()
            dictionary = json.loads(json_string)

        return collections.OrderedDict(sorted(dictionary.items()))

    except Exception as error: 
        log.warning("the json %s file containing the rotation axis locations is missing" % jfname)
        log.warning("to create one run:")
        log.warning("$ tomopy find_center --file-name %s" % top)
        # exit()


def auto_read_dxchange(params):
    log.info('  *** Auto parameter reading from DXchange file.')
    params = read_pixel_size(params)
    params = read_scintillator(params)
    params = read_rot_center(params)
    log.info('  *** *** Done')
    return params


def read_rot_center(params):
    """
    Read the rotation center from /process group in the DXchange file.
    Return: rotation center from this dataset or None if it doesn't exist.
    """
    log.info('  *** *** rotation axis')
    #First, try to read from the /process/tomopy-cli parameters
    with h5py.File(params.file_name, 'r') as file_name:
        try:
            dataset = '/process' + '/tomopy-cli-' + __version__ + '/' + 'find-rotation-axis' + '/'+ 'rotation-axis'
            params.rotation_axis = float(file_name[dataset][0])
            log.info('  *** *** Rotation center read from HDF5 file: {0:f}'.format(params.rotation_axis)) 
            return params
        except (KeyError, ValueError):
            log.warning('  *** *** No rotation center stored in the HDF5 file')
    #If we get here, we need to either find it automatically or from config file.
    log.warning('  *** *** No rotation axis stored in DXchange file')
    if (params.rotation_axis_auto == True):
        log.warning('  *** *** Auto axis location requested')
        log.warning('  *** *** Computing rotation axis')
        params.rotation_axis = find_center.find_rotation_axis(params) 
    log.info('  *** *** using config file value of {:f}'.format(params.rotation_axis))
    return params


def read_pixel_size(params):
    '''
    Read the pixel size and magnification from the DXchange file.
    Use to compute the effective pixel size.
    '''
    log.info('  *** auto pixel size reading')
    if params.pixel_size_auto != True:
        log.info('  *** *** OFF')
        return params
    pixel_size = config.param_from_dxchange(params.file_name,
                                            '/measurement/instrument/detector/pixel_size_x')
    mag = config.param_from_dxchange(params.file_name,
                                    '/measurement/instrument/detection_system/objective/magnification')
    #Handle case where something wasn't read right
    if not (pixel_size and mag):
        log.warning('  *** *** problem reading pixel size from DXchange')
        return params
    #What if pixel size isn't in microns, but in mm or m?
    for i in range(3):
        if pixel_size < 0.5:
            pixel_size *= 1e3
        else:
            break
    params.pixel_size = pixel_size / mag
    log.info('  *** *** effective pixel size = {:6.4e} microns'.format(params.pixel_size))
    return params


def read_scintillator(params):
    '''Read the scintillator type and thickness from DXchange.
    '''
    if params.scintillator_auto and params.beam_hardening_method.lower() == 'standard':
        log.info('  *** *** Find scintillator params from DXchange')
        params.scintillator_thickness = float(config.param_from_dxchange(params.file_name, 
                                            '/measurement/instrument/detection_system/scintillator/scintillating_thickness', 
                                            attr = None, scalar = True, char_array=False))
        log.info('  *** *** scintillator thickness = {:f}'.format(params.scintillator_thickness))
        scint_material_string = config.param_from_dxchange(params.file_name,
                                            '/measurement/instrument/detection_system/scintillator/description',
                                            scalar = False, char_array = True)
        if scint_material_string.lower().startswith('luag'):
            params.scintillator_material = 'LuAG_Ce'
        elif scint_material_string.lower().startswith('lyso'):
            params.scintillator_material = 'LYSO_Ce'
        elif scint_material_string.lower().startswith('yag'):
            params.scintillator_material = 'YAG_Ce' 
        else:
            log.warning('  *** *** scintillator {:s} not recognized!'.format(scint_material_string))
        log.warning('  *** *** using scintillator {:s}'.format(params.scintillator_material))
    #Run the initialization for beam hardening.  Needed in case rotation_axis must
    #be computed later.
    if params.beam_hardening_method.lower() == 'standard':
        beamhardening.initialize(params)
    return params 
