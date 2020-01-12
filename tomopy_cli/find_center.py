import os
import json
import tomopy
import dxchange

from tomopy_cli import log
from tomopy_cli import file_io
from tomopy_cli import prep


def find_rotation_axis(params):
    
    log.info("  *** calculating automatic center")
    data_size = file_io.get_dx_dims(params)
    ssino = int(data_size[1] * params.nsino)

    # Select sinogram range to reconstruct
    start = ssino
    end = start + 1
    sino = (start, end)

    # Read APS 32-BM raw data
    proj, flat, dark, theta = dxchange.read_aps_32id(params.hdf_file, sino=sino)
        
    # apply all preprocessing functions
    data = prep.all(proj, flat, dark, params)

    # find rotation center
    log.info("  *** find_center vo")
    rot_center = tomopy.find_center_vo(data)   
    log.info("  *** automatic center: %f" % rot_center)
    return rot_center


def auto(params):

    fname = params.hdf_file
    nsino = float(params.nsino)
    ra_fname = params.rotation_axis_file

    if os.path.isfile(fname):  
        rot_center = find_rotation_axis(params)
        
    elif os.path.isdir(fname):
        # Add a trailing slash if missing
        top = os.path.join(fname, '')

        # Set the file name that will store the rotation axis positions.
        jfname = top + ra_fname

        # log.info(os.listdir(top))
        h5_file_list = list(filter(lambda x: x.endswith(('.h5', '.hdf')), os.listdir(top)))
        h5_file_list.sort()

        log.info("Found: %s" % h5_file_list)
        log.info("Determining the rotation axis location ...")
        
        dic_centers = {}
        i=0
        for fname in h5_file_list:
            h5fname = top + fname
            params.hdf_file = h5fname
            rot_center = find_rotation_axis(params)
            params.hdf_file = top
            case =  {fname : rot_center}
            log.info(case)
            dic_centers[i] = case
            i += 1

        # Save json file containing the rotation axis
        json_dump = json.dumps(dic_centers)
        f = open(jfname,"w")
        f.write(json_dump)
        f.close()
        log.info("Rotation axis locations save in: %s" % jfname)

    else:
        log.info("Directory or File Name does not exist: %s " % fname)


    return

