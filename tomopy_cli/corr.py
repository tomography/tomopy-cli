import tomopy
import numpy as np

from tomopy_cli import log

def mask(data, params):
    
    if(params.reconstruction_mask):
        if 0 < params.reconstruction_mask_ratio <= 1:
            log.warning("  *** apply reconstruction mask ratio: %f " % params.reconstruction_mask_ratio)
            data = tomopy.circ_mask(data, axis=0, ratio=params.reconstruction_mask_ratio)
        else:
            log.error("  *** mask ratio must be between 0-1: %f is ignored" % params.reconstruction_mask_ratio)

    return data