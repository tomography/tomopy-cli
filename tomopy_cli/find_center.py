import os
import json

from tomopy_cli import config #, __version__
from tomopy_cli import log
from tomopy_cli import util


def auto(params):

    fname = params.hdf_file
    nsino = float(params.nsino)
    ra_fname = params.rotation_axis_file

    if os.path.isfile(fname):  
        rot_center = util.find_rotation_axis(fname, nsino)
        
    elif os.path.isdir(fname):
        # Add a trailing slash if missing
        top = os.path.join(fname, '')
        print(fname)
        print(top)
        print(ra_fname)
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
            rot_center = util.find_rotation_axis(h5fname, nsino)
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

        # update config file
        sections = config.FIND_CENTER_PARAMS
        config.write(params.config, args=params, sections=sections)
    
    else:
        log.info("Directory or File Name does not exist: %s " % fname)


    return

