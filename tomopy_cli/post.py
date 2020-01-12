import os
import tomopy
import numpy as np

from tomopy_cli import log


# this module will host post reconstuction data analysis (segmentation, etc.)

def segment(params):

    # slice/full reconstruction file location
    tail = os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_full_rec' + os.sep 
    top = os.path.dirname(params.hdf_file) + '_rec' + tail

    # log.info(os.listdir(top))
    rec_file_list = list(filter(lambda x: x.endswith(('.tiff', '.tif')), os.listdir(top)))
    rec_file_list.sort()


    log.info('found: %s' % rec_file_list)
    log.info('applying segmentation')

    log.error('not implemented')

