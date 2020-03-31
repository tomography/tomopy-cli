import os
import h5py
import json
import collections
import re
import tomopy
import dxchange
import dxchange.reader as dxreader
import dxfile.dxtomo as dx
import numpy as np

from tomopy_cli import log
from tomopy_cli import __version__
from tomopy_cli import find_center
from tomopy_cli import config
from tomopy_cli import beamhardening

 
def read_tomo(sino, params, ignore_flip = False):
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
    if (params.file_type == 'standard' or 
            (params.file_type == 'flip_and_stich' and ignore_flip)):
        # Read APS 32-BM raw data.
        log.info("  *** loading a stardard data set: %s" % params.file_name)
        proj, flat, dark, theta = _read_tomo(params, sino=sino)
    elif params.file_type == 'flip_and_stich':
        log.info("   *** loading a 360 deg flipped data set: %s" % params.file_name)
        proj360, flat360, dark360, theta360 = _read_tomo(params, sino=sino)
        proj, flat, dark = flip_and_stitch(params, proj360, flat360, dark360)
        theta = theta360[:len(theta360)//2] # take first half
    else: # params.file_type == 'mosaic':
        log.error("   *** loading a mosaic data set is not supported yet")
        exit()

    if params.reverse:
        log.info("  *** correcting for 180-0 data collection")
        step_size = (theta[1] - theta[0]) 
        theta_size = _read_theta_size(params)
        theta = np.linspace(np.pi, (0+step_size), theta_size)   

    proj, theta = blocked_view(proj, theta, params)
    proj, flat, dark = binning(proj, flat, dark, params)
    rotation_axis = params.rotation_axis / np.power(2, float(params.binning))
    log.info("  *** rotation center: %f" % rotation_axis)
    return proj, flat, dark, theta, rotation_axis


def _read_theta_size(params):
    if (str(params.file_format) in {'dx', 'aps2bm', 'aps7bm', 'aps32id'}):
        theta_size = dxreader.read_dx_dims(params.file_name, 'data')[0]
    else:
        log.error("  *** %s is not a supported file format" % params.file_format)
        exit()

    return theta_size


def _read_tomo(params, sino):

    if (str(params.file_format) in {'dx', 'aps2bm', 'aps7bm', 'aps32id'}):
        proj, flat, dark, theta = dxchange.read_aps_32id(params.file_name, sino=sino)
        log.info("  *** %s is a valid dx file format" % params.file_name)
        #Check if the flat and dark fields are single images or sets
        if len(flat.shape) == len(proj.shape):
            log.info('  *** median filter flat images')
            #Do a median filter on the first dimension
            flat = np.median(flat, axis=0, keepdims=True).astype(flat.dtype) 
        if len(dark.shape) == len(proj.shape):
            log.info('  *** median filter dark images')
            #Do a median filter on the first dimension
            dark = np.median(dark, axis=0, keepdims=True).astype(dark.dtype) 
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
    '''Code to take a 0-360 flip and stitch scan and provide stitched
    0-180 degree projections.
    '''
    num_stitched_angles = img360.shape[0]//2 
    new_width = int(2 * np.max([img360.shape[2] - params.rotation_axis_flip - 0.5,
                            params.rotation_axis_flip + 0.5]))
    log.info('  *** *** new image width = {:d}'.format(new_width))
    log.info('  *** *** rotation axis flip = {:f}'.format(params.rotation_axis_flip))
    img = np.zeros([num_stitched_angles,img360.shape[1], new_width],dtype=np.float32)
    flat = np.zeros([flat360.shape[0],flat360.shape[1], new_width],dtype=np.float32)
    dark = np.zeros([dark360.shape[0],dark360.shape[1], new_width],dtype=np.float32)
    # Just add both images, keeping an array to record whether there was an overlap
    weight = np.zeros((1,1,new_width))
    # Array to blend the overlap region smoothly between 0-180 and 180-360 degrees
    wedge = np.arange(img360.shape[2], 0, -1)
    # Take care of case where rotation axis is on the left edge of the image
    if params.rotation_axis_flip < img360.shape[2] - 1:
        img[:,:,:img360.shape[2]] = img360[num_stitched_angles:num_stitched_angles * 2,:,::-1] * wedge
        flat[:,:,:img360.shape[2]] = flat360[...,::-1] * wedge
        dark[:,:,:img360.shape[2]] = dark360[...,::-1] * wedge
        weight[0,0,:img360.shape[2]] += wedge
        img[:,:,-img360.shape[2]:] += img360[:num_stitched_angles,:,:] * wedge[::-1]
        flat[:,:,-img360.shape[2]:] += flat360 * wedge[::-1]
        dark[:,:,-img360.shape[2]:] += dark360 * wedge[::-1]
        weight[0,0,-img360.shape[2]:] += wedge[::-1]
    else:
        img[:,:,:img360.shape[2]] = img360[:num_stitched_angles,:,:] * wedge
        flat[:,:,:img360.shape[2]] = flat360 * wedge
        dark[:,:,:img360.shape[2]] = dark360 * wedge
        weight[0,0,:img360.shape[2]] += wedge
        img[:,:,-img360.shape[2]:] += img360[num_stitched_angles:num_stitched_angles * 2,:,::-1] * wedge[::-1]
        flat[:,:,-img360.shape[2]:] += flat360[...,::-1] * wedge[::-1]
        dark[:,:,-img360.shape[2]:] += dark360[...,::-1] * wedge[::-1]
        weight[-img360.shape[2]:] += wedge[::-1]

    # Divide through by the weight to take care of doubled regions
    img = (img / weight).astype(img360.dtype)
    flat = (flat / weight).astype(img360.dtype)
    dark = (dark / weight).astype(img360.dtype)

    params.rotation_axis = img.shape[2]/2 - 0.5

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
    log.info('  *** Auto parameter reading from the HDF file.')
    params = read_pixel_size(params)
    params = read_filter_materials(params)
    params = read_scintillator(params)
    params = read_bright_ratio(params)
    params = read_rot_center(params)
    log.info('  *** *** Done')
    return params


def read_rot_center(params):
    """
    Read the rotation center from /process group in the HDF file.
    Return: rotation center from this dataset or None if it doesn't exist.
    """
    log.info('  *** *** rotation axis')
    #Handle case of manual only: this is the easiest
    if params.rotation_axis_auto == 'manual':
        log.warning('  *** *** Force use of config file value = {:f}'.format(params.rotation_axis))
    elif params.rotation_axis_auto == 'auto':
        log.warning('  *** *** Force auto calculation without reading config value')
        log.warning('  *** *** Computing rotation axis')
        params = find_center.find_rotation_axis(params) 
    else:
        #Try to read from HDF5 file
        log.warning('  *** *** Try to read rotation center from file {:s}'.format(params.file_name))
        with h5py.File(params.file_name, 'r') as file_name:
            try:
                dataset = '/process' + '/tomopy-cli-' + __version__ + '/' + 'find-rotation-axis' + '/'+ 'rotation-axis'
                params.rotation_axis = float(file_name[dataset][0])
                dataset = '/process' + '/tomopy-cli-' + __version__ + '/' + 'find-rotation-axis' + '/'+ 'rotation-axis-flip'
                params.rotation_axis_flip = float(file_name[dataset][0])
                log.info('  *** *** Rotation center read from HDF5 file: {0:f}'.format(params.rotation_axis)) 
                log.info('  *** *** Rotation center flip read from HDF5 file: {0:f}'.format(params.rotation_axis_flip)) 
                return params
            except (KeyError, ValueError):
                log.warning('  *** *** No rotation center stored in the HDF5 file')
        #If we get here, we need to either find it automatically or from config file.
        log.warning('  *** *** No rotation axis stored in the HDF file')
        if (params.rotation_axis_auto == 'read_auto'):
            log.warning('  *** *** fall back to auto calculation')
            log.warning('  *** *** Computing rotation axis')
            params = find_center.find_rotation_axis(params) 
        else:
            log.info('  *** *** using config file value of {:f}'.format(params.rotation_axis))
    return params


def read_filter_materials(params):
    '''Read the beam filter configuration from the HDF file.
    
    If params.filter_1_material and/or params.filter_2_material are
    'auto', then try to read the filter configuration recorded during
    acquisition in the HDF5 file.
    
    Parameters
    ==========
    params
    
      The global parameter object, should have *filter_1_material*,
      *filter_1_thickness*, *filter_2_material*, and
      *filter_2_thickness* attributes.
    
    Returns
    =======
    params
      An equivalent object to the *params* input, optionally with
      *filter_1_material*, *filter_1_thickness*, *filter_2_material*,
      and *filter_2_thickness* attributes modified to reflect the HDF5
      file.
    
    '''
    log.info('  *** auto reading filter configuration')
    # Read the relevant data from disk
    filter_path = '/measurement/instrument/filters/Filter_{idx}_Material'
    param_path = 'filter_{idx}_{attr}'
    for idx_filter in (1, 2):
        filter_param = getattr(params, param_path.format(idx=idx_filter, attr='material'))
        if filter_param == 'auto':
            # Read recorded filter condition from the HDF5 file
            filter_str = config.param_from_dxchange(params.file_name,
                                                    filter_path.format(idx=idx_filter),
                                                    char_array=True, scalar=False)
            if filter_str is None:
                log.warning('  *** *** Could not load filter %d configuration from HDF5 file.' % idx_filter)
                material, thickness = _filter_str_to_params('Open')
            else:
                material, thickness = _filter_str_to_params(filter_str)
            # Update the params with the loaded values
            setattr(params, param_path.format(idx=idx_filter, attr='material'), material)
            setattr(params, param_path.format(idx=idx_filter, attr='thickness'), thickness)
            log.info('  *** *** Filter %d: (%s %f)' % (idx_filter, material, thickness))
    return params


def _filter_str_to_params(filter_str):
    # Any material with zero thickness is equivalent to being open
    open_filter = ('Al', 0.)
    if filter_str == 'Open':
        # No filter is installed
        material, thickness = open_filter
    else:
        # Parse the filter string to get the parameters
        filter_re = '(?P<material>[A-Za-z_]+)_(?P<thickness>[0-9.]+)(?P<unit>[a-z]*)'
        match = re.match(filter_re, filter_str)
        if match:
            material, thickness, unit = match.groups()
        else:
            log.warning('  *** *** Cannot interpret filter "%s"' % filter_str)
            material, thickness = open_filter
            unit = 'um'
        # Convert strings into numbers
        thickness = float(thickness)
        factors = {
            'nm': 1e-3,
            'um': 1,
            'mm': 1e3,
        }
        try:
            factor = factors[unit]
        except KeyError:
            log.warning('  *** *** Cannot interpret filter unit in "%s"' % filter_str)
            factor = 1
        thickness *= factor
    return material, thickness


def read_pixel_size(params):
    '''
    Read the pixel size and magnification from the HDF file.
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
        log.warning('  *** *** problem reading pixel size from the HDF file')
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
    '''Read the scintillator type and thickness from the HDF file.
    '''
    if params.scintillator_auto and params.beam_hardening_method.lower() == 'standard':
        log.info('  *** auto reading scintillator params')
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
        log.info('  *** *** using scintillator {:s}'.format(params.scintillator_material))
    #Run the initialization for beam hardening.  Needed in case rotation_axis must
    #be computed later.
    if params.beam_hardening_method.lower() == 'standard':
        beamhardening.initialize(params)
    return params 


def read_bright_ratio(params):
    '''Read the ratio between the bright exposure and other exposures.
    '''
    log.info('  *** *** %s' % params.flat_correction_method)
    if params.scintillator_auto and params.flat_correction_method == 'standard':
        log.info('  *** *** Find bright exposure ratio params from the HDF file')
        try:
            bright_exp = config.param_from_dxchange(params.file_name,
                                        '/measurement/instrument/detector/brightfield_exposure_time',
                                        attr = None, scalar = True, char_array = False)
            log.info('  *** *** %f' % bright_exp)
            norm_exp = config.param_from_dxchange(params.file_name,
                                        '/measurement/instrument/detector/exposure_time',
                                        attr = None, scalar = True, char_array = False)
            log.info('  *** *** %f' % norm_exp)
            params.bright_exp_ratio = bright_exp / norm_exp
            log.info('  *** *** found bright exposure ratio of {0:6.4f}'.format(params.bright_exp_ratio))
        except:
            log.warning('  *** *** problem getting bright exposure ratio.  Use 1.')
            params.bright_exp_ratio = 1
    else:
            params.bright_exp_ratio = 1
    return params


def convert(params):

    head_tail = os.path.split(params.old_projection_file_name)

    new_hdf_file_name = head_tail[0] + os.sep + os.path.splitext(head_tail[1])[0] + '.h5'

    print('converting data file: %s in new format: %s' % (params.old_projection_file_name, new_hdf_file_name))
    print('using %s as dark and %s as white field' %(params.old_dark_file_name, params.old_white_file_name))
    exchange_base = "exchange"

    tomo_grp = '/'.join([exchange_base, 'data'])
    flat_grp = '/'.join([exchange_base, 'data_white'])
    dark_grp = '/'.join([exchange_base, 'data_dark'])
    theta_grp = '/'.join([exchange_base, 'theta'])
    tomo = dxreader.read_hdf5(params.old_projection_file_name, tomo_grp)
    flat = dxreader.read_hdf5(params.old_white_file_name, flat_grp)
    dark = dxreader.read_hdf5(params.old_dark_file_name, dark_grp)
    theta = dxreader.read_hdf5(params.old_projection_file_name, theta_grp)

    # Open DataExchange file
    f = dx.File(new_hdf_file_name, mode='w') 

    f.add_entry(dx.Entry.data(data={'value': tomo, 'units':'counts'}))
    f.add_entry(dx.Entry.data(data_white={'value': flat, 'units':'counts'}))
    f.add_entry(dx.Entry.data(data_dark={'value': dark, 'units':'counts'}))
    f.add_entry(dx.Entry.data(theta={'value': theta, 'units':'degrees'}))

    f.close()
