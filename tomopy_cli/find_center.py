import os
import json
import tomopy
import numpy as np
import h5py

from tomopy_cli import log
from tomopy_cli import prep
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

    # Select sinogram range to reconstruct
    sino_start = ssino
    sino_end = sino_start + pow(2, int(params.binning)) 

    sino = (int(sino_start), int(sino_end))

    # Read APS 32-BM raw data
    proj, flat, dark, theta, params_rotation_axis_ignored = file_io.read_tomo(sino, params)
        
    # apply all preprocessing functions
    data = prep.all(proj, flat, dark, params, sino)

    # find rotation center
    log.info("  *** find_center vo")
    rot_center = tomopy.find_center_vo(data)   
    log.info("  *** automatic center: %f" % rot_center)

    return rot_center * np.power(2, float(params.binning))
