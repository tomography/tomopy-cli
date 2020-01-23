import os
import tomopy
import numpy as np

from tomopy_cli import log
logger = log.logger


# this module will host post reconstuction data analysis (segmentation, etc.)

def segment(params):

    # slice/full reconstruction file location
    tail = os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_rec' + os.sep 
    top = os.path.dirname(params.hdf_file) + '_rec' + tail

    # logger.info(os.listdir(top))
    if os.path.isdir(top):
        rec_file_list = list(filter(lambda x: x.endswith(('.tiff', '.tif')), os.listdir(top)))
        rec_file_list.sort()


        logger.info('found in %s' % top)
        logger.info('files %s' % rec_file_list)
        logger.info('applying segmentation')
        logger.warning('not implemented')
    else:
        logger.error("ERROR: the directory %s does not exist" % top)
        logger.error("ERROR: to create one run a full reconstruction first:")
        logger.error("ERROR: $ tomopy recon --reconstruction-type full --hdf-file %s" % params.hdf_file)

