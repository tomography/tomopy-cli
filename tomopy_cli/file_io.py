import os
import logging
from pathlib import Path
import collections
import re
from typing import List

import h5py
import tomopy
import dxchange
import dxchange.reader as dxreader
import dxfile.dxtomo as dx
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import yaml

from tomopy_cli import __version__
from tomopy_cli import find_center
from tomopy_cli import config
from tomopy_cli import beamhardening

__author__ = "Francesco De Carlo, Viktor Nikitin, Alan Kastengren, Mark Wolfman"
__credits__ = "Pavel Shevchenko"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['read_tomo', 'blocked_view', 'binning', 'flip_and_stitch', 'patch_projection', 
           'get_dx_dims', 'file_base_name', 'path_base_name', 'auto_read_dxchange', 'read_rot_center', 
           'read_filter_materials', 'read_filter_materials_tomoscan', 'read_pixel_size', 
           'read_scintillator', 'read_bright_ratio', 'check_item_exists_hdf', 'convert', 
           'write_hdf5', 'yaml_file_list']


log = logging.getLogger(__name__)


def read_tomo(sino, proj, params, ignore_flip = False):
    """
    Read in the tomography data.

    Parameters
    ----------
    sino : tuple of (start_row, end_row) rows to be read in
    proj : tuple of (start_proj, end_proj) projections to be read in
    
    params : parameters for reconstruction
    
    Returns
    -------
    ndarray
        3D tomographic data.
    ndarray
        3D flat field data.
    ndarray
        3D dark field data.
    ndarray
        1D theta in radian.
    float
        location of the rotation axis
    """
    if (params.file_type == 'standard' or params.file_type == 'double_fov' or
            (params.file_type == 'flip_and_stich' and ignore_flip)):
        # Read APS 32-BM raw data.
        log.info("  *** loading a stardard data set: %s" % params.file_name)
        proj, flat, dark, theta = _read_tomo(params, sino=sino, proj=proj)
    elif params.file_type == 'flip_and_stich':
        log.info("   *** loading a 360 deg flipped data set: %s" % params.file_name)
        proj360, flat360, dark360, theta360 = _read_tomo(params, sino=sino, proj=proj)
        proj, flat, dark, theta = flip_and_stitch(params, proj360, flat360, dark360, theta360)
    else: # params.file_type == 'mosaic':
        log.error("   *** loading a mosaic data set is not supported yet")
        exit()

    if params.correct_camera_nonlinearity:
        log.info("  *** correcting camera nonlinearity in flat fields")
        flat = camera_nonlinearity_correct(flat, params)
        log.info("  *** correcting camera nonlinearity in dark fields")
        dark = camera_nonlinearity_correct(dark, params)
        log.info("  *** correcting camera nonlinearity in projection fields")
        proj = camera_nonlinearity_correct(proj, params)

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


def _read_tomo(params, sino, proj):

    if (str(params.file_format) in {'dx', 'aps2bm', 'aps7bm', 'aps32id'}):
        # temporary work around
        #flat_file = '/local/data/2020-10/PazPuente/flat_samp2_417.h5'
        #print('flat fields are taken from:', flat_file)
#        proj_bad, flat, dark, theta_bad = dxchange.read_aps_32id(flat_file, sino=sino)
#        proj, flat_bad, dark_bad, theta = dxchange.read_aps_32id(params.file_name, sino=sino)
        proj, flat, dark, theta = dxchange.read_aps_32id(params.file_name, sino=sino, proj=proj)
        log.info("  *** %s is a valid dx file format" % params.file_name)
        # Check if the flat and dark fields are single images or sets
        if len(flat.shape) == len(proj.shape):
            log.info('  *** median filter flat images')
            # Do a median filter on the first dimension
            flat = np.median(flat, axis=0, keepdims=True).astype(flat.dtype) 
        if dark is None:
            dark = np.zeros_like(proj[0,...])
        if len(dark.shape) == len(proj.shape):
            log.info('  *** median filter dark images')
            # Do a median filter on the first dimension
            dark = np.median(dark, axis=0, keepdims=True).astype(dark.dtype) 
    else:
        log.error("  *** %s is not a supported file format" % params.file_format)
        exit()
    return proj, flat, dark, theta


def blocked_view(proj, theta, params):
    log.info("  *** correcting for blocked view data collection")
    if params.blocked_views:
        log.warning('  *** *** ON')
        # miss_angles = [params.blocked_views_start, params.blocked_views_end]
        # # Manage the missing angles:
        # proj = np.concatenate((proj[0:miss_angles[0],:,:], proj[miss_angles[1]:,:,:]), axis=0)
        # theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]:]))
         
        # easier managing of missing angles: Viktor
        st = params.blocked_views_start
        end = params.blocked_views_end
        log.warning('%f %f',st,end)
        ids = np.where(((theta)%np.pi<st) + ((theta-st)%np.pi>end-st))[0]
        proj = proj[ids]
        theta = theta[ids]
        print(theta)

    else:
        log.warning('  *** *** OFF')

    return proj, theta


def binning(proj, flat, dark, params):
    """
    Bin the tomography data.

    Parameters
    ----------
    proj : projection data, 3D Numpy array
    flat : projection flatfield data, 2D Numpy array
    
    dark : projection dark field data, 2D Numpy array
    
    params : parameters for reconstruction
    
    Returns
    -------
    ndarray
        3D binned projection data.
    ndarray
        2D binned flat field data.
    ndarray
        2D binned dark field data.
    """
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


def flip_and_stitch(params, img360, flat360, dark360, theta360):
    """
    Stitch together data for flip-and-stitch (0-360 degree offset center) scan.

    Parameters
    ----------
    params : dict of reconstruction parameters
    img360 : projection data from 0-360 degrees, 3D Numpy array
    flat360 : flatfield data, 2D Numpy array
    
    dark360 : dark field data, 2D Numpy array
    
    params : parameters for reconstruction
    
    Returns
    -------
    ndarray
        3D binned projection data stitched together, 0-180 degree domain
    ndarray
        2D binned flat field data stitched together, 0-180 degree domain
    ndarray
        2D binned dark field data stitched together, 0-180 degree domain
    ndarray
        1D array of valid theta values, 0-180 degree domain
    """
    theta_0_180, good_0_180, good_180_360 = reconcile_flip_and_stitch_angles(theta360) 
    num_stitched_angles = np.sum(good_0_180)
    new_width = int(2 * np.max([img360.shape[2] - params.rotation_axis_flip - 0.5,
                            params.rotation_axis_flip + 0.5]))
    log.info('  *** *** new image width = {:d}, rotation_axis_flip = {:f}'.format(
                new_width, params.rotation_axis_flip))
    img = np.zeros([num_stitched_angles,img360.shape[1], new_width],dtype=np.float32)
    flat = np.zeros([flat360.shape[0],flat360.shape[1], new_width],dtype=np.float32)
    dark = np.zeros([dark360.shape[0],dark360.shape[1], new_width],dtype=np.float32)
    # Just add both images, keeping an array to record whether there was an overlap
    weight = np.zeros((1,1,new_width))
    # Array to blend the overlap region smoothly between 0-180 and 180-360 degrees
    wedge = np.arange(img360.shape[2], 0, -1)
    # Take care of case where rotation axis is on the left edge of the image
    if params.rotation_axis_flip < img360.shape[2] - 1:
        img[:,:,:img360.shape[2]] = img360[good_180_360,:,::-1] * wedge
        flat[:,:,:img360.shape[2]] = flat360[...,::-1] * wedge
        dark[:,:,:img360.shape[2]] = dark360[...,::-1] * wedge
        weight[0,0,:img360.shape[2]] += wedge
        img[:,:,-img360.shape[2]:] += img360[good_0_180,:,:] * wedge[::-1]
        flat[:,:,-img360.shape[2]:] += flat360 * wedge[::-1]
        dark[:,:,-img360.shape[2]:] += dark360 * wedge[::-1]
        weight[0,0,-img360.shape[2]:] += wedge[::-1]
    else:
        img[:,:,:img360.shape[2]] = img360[good_0_180,:,:] * wedge
        flat[:,:,:img360.shape[2]] = flat360 * wedge
        dark[:,:,:img360.shape[2]] = dark360 * wedge
        weight[0,0,:img360.shape[2]] += wedge
        img[:,:,-img360.shape[2]:] += img360[good_180_360,:,::-1] * wedge[::-1]
        flat[:,:,-img360.shape[2]:] += flat360[...,::-1] * wedge[::-1]
        dark[:,:,-img360.shape[2]:] += dark360[...,::-1] * wedge[::-1]
        weight[-img360.shape[2]:] += wedge[::-1]

    # Divide through by the weight to take care of doubled regions
    img = (img / weight).astype(img360.dtype)
    flat = (flat / weight).astype(img360.dtype)
    dark = (dark / weight).astype(img360.dtype)

    params.rotation_axis = img.shape[2]/2 - 0.5

    return img, flat, dark, theta_0_180


def reconcile_flip_and_stitch_angles(theta360):
    """
    Figure out the valid angles where we have images from both 0-180 and 180-360
    This is necessary in case their are missing angles.
    """
    theta360deg = np.degrees(theta360)
    good_180_360 = np.full((theta360deg.shape[0]), False, dtype=np.bool_)
    good_0_180 = np.full((theta360deg.shape[0]), False, dtype=np.bool_)
    #Loop across the entries from 0 to 180 degrees from the start point
    for i in theta360deg[(theta360deg - theta360deg[0]) <= 180]:
        #If there is a match between 0-180 and 180-360, indicate this is good
        comparison_array = np.abs(theta360deg - i - 180.)
        if np.min(comparison_array) < 0.001:
            good_0_180[np.argmin(np.abs(theta360deg - i))] = True
            good_180_360[np.argmin(comparison_array)] = True
    return theta360[good_0_180], good_0_180, good_180_360

            
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
    # Handle case of manual only: this is the easiest
    if params.rotation_axis_auto == 'manual':
        log.info('  *** *** Force use of config file value = {:f}'.format(params.rotation_axis))
    elif params.rotation_axis_auto == 'auto':
        log.info('  *** *** Force auto calculation without reading config value')
        log.info('  *** *** Computing rotation axis')
        params = find_center.find_rotation_axis(params)
    else:
        # Try to read from HDF5 file
        log.info('  *** *** Try to read rotation center from file {}'.format(params.file_name))
        with h5py.File(params.file_name, 'r') as file_name:
            try:
                dataset = '/process/tomopy-cli-{}/find-rotation-axis/rotation-axis'.format(__version__)
                params.rotation_axis = float(file_name[dataset][0])
                dataset = '/process/tomopy-cli-{}/find-rotation-axis/rotation-axis-flip'.format(__version__)
                params.rotation_axis_flip = float(file_name[dataset][0])
                log.info('  *** *** Rotation center read from HDF5 file: {0:f}'.format(params.rotation_axis)) 
                log.info('  *** *** Rotation center flip read from HDF5 file: {0:f}'.format(params.rotation_axis_flip))
                return params
            except (KeyError, ValueError):
                log.warning('  *** *** No rotation center stored in the HDF5 file')
        # If we get here, we need to either find it automatically or from config file.
        log.warning('  *** *** No rotation axis stored in the HDF file')
        if (params.rotation_axis_auto == 'read_auto'):
            log.warning('  *** *** fall back to auto calculation')
            log.warning('  *** *** Computing rotation axis')
            params = find_center.find_rotation_axis(params) 
        else:
            log.warning('  *** *** using config file value of {:f}'.format(params.rotation_axis))
    return params


def read_filter_materials(params):
    '''Read the beam filter configuration.
    This discriminates between files created with tomoScan and
    the previous meta data format.
    '''
    if check_item_exists_hdf(params.file_name, '/measurement/instrument/attenuator_1'):
        return read_filter_materials_tomoscan(params)
    else:
        return read_filter_materials_old(params)


def read_filter_materials_tomoscan(params):
    '''Read the beam filter configuration from the HDF file.
    
    If params.filter_{n}_auto for n in [1,2,3] is True,
    then try to read the filter configuration recorded during
    acquisition in the HDF5 file.
    
    Parameters
    ==========
    params
    
      The global parameter object, should have *filter_n_material*,
      *filter_n_thickness*, and *filter_n_auto* for n in [1,2,3]
    
    Returns
    =======
    params
      An equivalent object to the *params* input, optionally with
      *filter_n_material* and *filter_n_thickness*
      attributes modified to reflect the HDF5 file.
    
    '''
    log.info('  *** auto reading filter configuration')
    # Read the relevant data from disk
    filter_path = '/measurement/instrument/attenuator_{idx}'
    param_path = 'filter_{idx}_{attr}'
    for idx_filter in range(1,4,1):
        if not check_item_exists_hdf(params.file_name, filter_path.format(idx = idx_filter)):
            log.warning('  *** *** Filter {idx} not found in HDF file.  Set this filter to none'
                                    .format(idx = idx_filter))
            setattr(params, param_path.format(idx=idx_filter, attr='material'), 'Al')
            setattr(params, param_path.format(idx=idx_filter, attr='thickness'), 0.0)
            continue
        filter_auto = getattr(params, param_path.format(idx=idx_filter, attr='auto'))
        if filter_auto != 'True' and filter_auto != True:
            log.warning('  *** *** do not auto read filter {n}'.format(n=idx_filter))
            continue
        log.warning('  *** *** auto reading parameters for filter {0}'.format(idx_filter))
        # See if there are description and thickness fields
        if check_item_exists_hdf(params.file_name, filter_path.format(idx = idx_filter) + '/description'):
            filt_material = config.param_from_dxchange(params.file_name,
                                        filter_path.format(idx=idx_filter) + '/description',
                                        char_array = True, scalar = False)
            filt_thickness = int(config.param_from_dxchange(params.file_name,
                                        filter_path.format(idx=idx_filter) + '/thickness',
                                        char_array = False, scalar = True))
        else:
            #The filter info is just the raw string from the filter unit.
            log.warning('  *** *** filter {idx} info must be read from the raw string'
                            .format(idx = idx_filter))
            filter_str = config.param_from_dxchange(params.file_name,
                                        filter_path.format(idx=idx_filter) + '/setup/filter_unit_text',
                                        char_array = True, scalar = False)
            if filter_str is None:
                log.warning('  *** *** Could not load filter %d configuration from HDF5 file.' % idx_filter)
                filt_material, filt_thickness = _filter_str_to_params('Open')
            else: 
                filt_material, filt_thickness = _filter_str_to_params(filter_str)

        # Update the params with the loaded values
        setattr(params, param_path.format(idx=idx_filter, attr='material'), filt_material)
        setattr(params, param_path.format(idx=idx_filter, attr='thickness'), filt_thickness)
        log.info('  *** *** Filter %d: (%s %f)' % (idx_filter, filt_material, filt_thickness))
    return params


def read_filter_materials_old(params):
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
    
    if check_item_exists_hdf(params.file_name,
                                '/measurement/instrument/detection_system/objective/resolution'):
        params.pixel_size = config.param_from_dxchange(params.file_name,
                                            '/measurement/instrument/detection_system/objective/resolution')
        log.info('  *** *** effective pixel size = {:6.4e} microns'.format(params.pixel_size))
        return(params)
    if check_item_exists_hdf(params.file_name,
                                '/measurement/instrument/detector/actual_pixel_size_x'):
        params.pixel_size = config.param_from_dxchange(params.file_name,
                                            '/measurement/instrument/detector/actual_pixel_size_x')
        log.info('  *** *** effective pixel size = {:6.4e} microns'.format(params.pixel_size))
        return(params)
    log.warning('  *** tomoScan resolution parameter not found.  Try old format')
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
    if params.scintillator_auto:
        log.info('  *** auto reading scintillator params')
        possible_names = ['/measurement/instrument/detection_system/scintillator/scintillating_thickness',
                        '/measurement/instrument/detection_system/scintillator/active_thickness']
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                val = config.param_from_dxchange(params.file_name,
                                         pn, attr=None,
                                         scalar=True,
                                         char_array=False)
                params.scintillator_thickness = float(val)
                break
        log.info('  *** *** scintillator thickness = {:f}'.format(params.scintillator_thickness))
        possible_names = ['/measurement/instrument/detection_system/scintillator/name',
                        '/measurement/instrument/detection_system/scintillator/type',
                        '/measurement/instrument/detection_system/scintillator/description']
        scint_material_string = ''
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                scint_material_string = config.param_from_dxchange(params.file_name,
                                            pn, scalar = False, char_array = True)
                break
        else:
            log.warning('  *** *** no scintillator material found')
            return(params)
        if scint_material_string.lower().startswith('luag'):
            params.scintillator_material = 'LuAG_Ce'
        elif scint_material_string.lower().startswith('lyso'):
            params.scintillator_material = 'LYSO_Ce'
        elif scint_material_string.lower().startswith('yag'):
            params.scintillator_material = 'YAG_Ce' 
        else:
            log.warning('  *** *** scintillator {:s} not recognized!'.format(scint_material_string))
        log.info('  *** *** using scintillator {:s}'.format(params.scintillator_material))
    return params 


def read_bright_ratio(params):
    '''Read the ratio between the bright exposure and other exposures.
    '''
    log.info('  *** *** %s' % params.flat_correction_method)
    if params.flat_correction_method != 'standard' or (not params.scintillator_auto):
        log.warning('  *** *** skip finding exposure ratio')
        params.bright_exp_ratio = 1
        return params
    log.info('  *** *** Find bright exposure ratio params from the HDF file')
    try:
        possible_names = ['/measurement/instrument/detector/different_flat_exposure',
                        '/process/acquisition/flat_fields/different_flat_exposure']
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                diff_bright_exp = config.param_from_dxchange(params.file_name, pn,
                                    attr = None, scalar = False, char_array = True)
                break
        if diff_bright_exp.lower() == 'same':
            log.error('  *** *** used same flat and data exposures')
            params.bright_exp_ratio = 1
            return params
        possible_names = ['/measurement/instrument/detector/exposure_time_flat',
                        '/process/acquisition/flat_fields/flat_exposure_time',
                        '/measurement/instrument/detector/brightfield_exposure_time']
        for pn in possible_names:
            if check_item_exists_hdf(params.file_name, pn):
                bright_exp = config.param_from_dxchange(params.file_name, pn,
                                    attr = None, scalar = True, char_array = False)
                break    
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
    return params


def check_item_exists_hdf(hdf_filename, item_name):
    '''Checks if an item exists in an HDF file.
    Inputs
    hdf_filename: str filename or pathlib.Path object for HDF file to check
    item_name: name of item whose existence needs to be checked
    path: str path to check.  Default to None
    '''
    with h5py.File(hdf_filename, 'r') as hdf_file:
        return item_name in hdf_file


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


def write_hdf5(data, fname, dname='volume', dtype=None,
               dest_idx=None, maxsize=None, overwrite=False):
    """Write data to hdf5 file in a specific dataset.
    
    This function supports partial writing of data through af
    combination of *maxsize* and *dest_idx* options. For example, to
    write slices 10 to 16 of a (32, 32, 32) volume::
    
        assert data.shape == (32, 32, 32)
        file_io.write_hdf5(data[10:16], maxsize=data.shape, dest_idx=slice(10,16), ...)
    
    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        File name to which the data is saved. ``.h5`` extension
        will be appended if it does not already have one.
    dname : str, optional
        Name for dataset where data will be written.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    dest_idx : optional
        A valid index for the dataset such that ``dataset[target_idx]
        = data`` will properly write the data to the dataset.
    maxsize : int, optional
        Maximum size that the dataset can be resized to along the
        given axis.
    
    """
    # Extract default values if not given
    if maxsize is None:
        maxsize = data.shape
    if dtype is None:
        dtype = data.dtype
    if dest_idx is None:
        dest_idx = ()
    # Create parent directory if necessary
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    # Open the HDF5 file so we can save data to it
    with h5py.File(fname, mode='a') as h5fp:
        # Delete the dataset if it already exists and is being overwritten
        if dname in h5fp.keys() and overwrite:
            del h5fp[dname]
        # Create a new dataset if necessary
        try:
            ds = h5fp.require_dataset(dname, shape=maxsize, dtype=dtype, fillvalue=np.nan, exact=True)
        except TypeError as e:
            msg = str(e) + ". Use *overwrite=True* to overwrite existing dataset."
            raise type(e)(msg)
        # Save the data
        ds[dest_idx] = data


def yaml_file_list(file_path: Path)->List[Path]:
    """Open a YAML file and return the list of files within.
    This function does not parse the parameters contained inside,
    merely returns a list of the files that are referenced. For
    updating parameters on a per-file basis, use
    ``config.yaml_args()``.
    Parameters
    ==========
    file_path
      A pathlib Path object pointing to the file to open.
    Returns
    =======
    file_list
      The list of file names found. There is no guarantee that these
      files are suitable for reconsturction, or even exist at all.
    """
    with open(file_path, mode='r') as fp:
        yaml_data = yaml.safe_load(fp.read())
    file_list = [Path(k) for k in yaml_data.keys()]
    return file_list


def camera_nonlinearity_correct(data, params):
    '''Corrects for nonlinearity in the camera
    Takes data from params to form a spline fit, then uses the spline
    fit to correct for the camera nonlinearity.
    Inputs:
    data: numpy array of camera data
    params: parameters from config file and command line
    '''
    log.info('*** correcting for camera nonlinearity')
    #Parse the params.camera_signal and params.corrected_signal
    camera_signal = np.array([float(i) for i in params.camera_signal.split(",")])
    corrected_signal = np.array([float(i) for i in params.corrected_signal.split(",")])
    knots = np.linspace(camera_signal[1], camera_signal[-2], len(camera_signal)//2)
    spline_obj = LSQUnivariateSpline(
                                    camera_signal,
                                    corrected_signal,
                                    knots,
                                    check_finite = True)
    temp = spline_obj(data)
    temp[temp > np.iinfo(data.dtype).max] = np.iinfo(data.dtype).max
    temp[temp < 0] = 0
    return temp.astype(data.dtype)
