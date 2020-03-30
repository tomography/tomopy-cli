import os
import json
import tomopy
import numpy as np
import h5py
from skimage.filters import gaussian
import skimage.feature

from tomopy_cli import log
from tomopy_cli import prep
from tomopy_cli import config
from tomopy_cli import file_io


def find_rotation_axis(params):

    fname = params.file_name
    ra_fname = params.rotation_axis_file

    if os.path.isfile(fname):  
        return _find_rotation_axis(params)
        
    elif os.path.isdir(fname):
        # Add a trailing slash if missing
        top = os.path.join(fname, '')

        # log.info(os.listdir(top))
        h5_file_list = list(filter(lambda x: x.endswith(('.h5', '.hdf')), os.listdir(top)))
        h5_file_list.sort()

        log.info("Found: %s" % h5_file_list)
        log.info("Determining the rotation axis location")
        
        dic_centers = {}
        i=0
        for fname in h5_file_list:
            h5fname = top + fname
            params.file_name = h5fname
            rot_center = _find_rotation_axis(params)
            params.file_name = top
            case =  {fname : rot_center}
            log.info("  *** file: %s; rotation axis %f" % (fname, rot_center))
            dic_centers[i] = case
            i += 1

        # Set the json file name that will store the rotation axis positions.
        jfname = top + ra_fname
        # Save json file containing the rotation axis
        json_dump = json.dumps(dic_centers)
        f = open(jfname,"w")
        f.write(json_dump)
        f.close()
        log.info("Rotation axis locations save in: %s" % jfname)

    else:
        log.info("Directory or File Name does not exist: %s " % fname)


def _find_rotation_axis(params):
    
    log.info("  *** calculating automatic center")
    data_size = file_io.get_dx_dims(params)
    ssino = int(data_size[1] * params.nsino)
    params = file_io.read_pixel_size(params)
    params = file_io.read_filter_materials(params)
    params = file_io.read_scintillator(params)
    params = file_io.read_bright_ratio(params)

    # Select sinogram range to reconstruct
    sino_start = ssino
    sino_end = sino_start + pow(2, int(params.binning)) 

    sino = (int(sino_start), int(sino_end))

    # Read APS 32-BM raw data
    proj, flat, dark, theta, params_rotation_axis_ignored = file_io.read_tomo(sino, params, True)
        
    # apply all preprocessing functions
    data = prep.all(proj, flat, dark, params, sino)

    # if flip and stitch, just use the overlapped part of the dataset
    if params.file_type == 'flip_and_stich':
        params = _find_rotation_axis_flip_stitch(data, params)
    else:        
        # find rotation center
        log.info("  *** find_center vo")
        # if we start at 0 and end at 180, remove last angle
        if np.isclose(theta[-1] - theta[0], np.pi, 1e-4):
            data = data[:-1,...]
        params.rotation_axis = tomopy.find_center_vo(data) * np.power(2, float(params.binning))
        params.rotation_axis_flip = -1
    log.info("  *** automatic center: %f" % params.rotation_axis)
    return params


def _find_rotation_axis_flip_stitch(data, params):
    '''Code to find the center of rotation for a flip-and-stitch scan.
    Unlike for 0-180 degree scans, we have images from two angles
    180 degrees apart to compare in the region viewed at all angles.
    '''
    log.info('  *** *** finding rotation axis for flip-and-stitch scan')
    log.info(data.shape)
    #Make images of the two halves of the sinogram
    #Only use the part near the rotation_axis_flip
    log.info('  *** *** using overlap area, original rotation-axis-flip = {0:f}'
                .format(params.rotation_axis_flip))
    column_slice = None
    if params.rotation_axis_flip < data.shape[2]//2:
        column_slice = slice(None, int(params.rotation_axis_flip * 2 + 1), 1)
    else:
        subset_size = int((data.shape[2] - params.rotation_axis_flip) * 2) - 1
        column_slice = slice(-subset_size, None, 1)
    half_num_angles = data.shape[0]//2
    img_0_180 = data[:half_num_angles,0,column_slice]
    img_180_360 = data[half_num_angles:2 * half_num_angles,0,column_slice]
    img_180_360 = np.flip(img_180_360, axis=1)
    log.info('  *** *** shape of images to correlate is ({0:d}, {1:d})'
                .format(*img_0_180.shape)) 
    #Do an unsharp mask on these to get only the fine features and zero mean
    img_0_180 -= skimage.filters.gaussian(img_0_180, sigma=10, mode='reflect')
    img_180_360 -= skimage.filters.gaussian(img_180_360, sigma=10, mode='reflect')
    correlation_matrix = skimage.feature.match_template(img_0_180, img_180_360, pad_input=True)
    match_location = np.argmax(correlation_matrix[half_num_angles//2,:])
    axis_shift = (match_location - params.rotation_axis_flip) / 2.0
    log.info('  *** *** match location = {:d}'.format(match_location))
    log.info('  *** *** axis shift = {:f}'.format(axis_shift))
    params.rotation_axis_flip += axis_shift
    new_size = data.shape[2] + np.abs(axis_shift) * 2.0
    params.rotation_axis = new_size / 2 - 0.5
    log.info('  *** *** rotation axis before stitch = {:f}'.format(params.rotation_axis_flip))
    log.info('  *** *** rotation axis = {:f}'.format(params.rotation_axis))
    return params
