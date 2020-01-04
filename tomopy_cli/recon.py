import os
import logging
import glob
import tempfile
import sys
import numpy as np
import tomopy
import dxchange

from tomopy_cli import config #, __version__
from tomopy_cli import log


def tomo(params):

    # update config file
    sections = config.RECON_PARAMS
    config.write(params.config, args=params, sections=sections)
    return

    fname = str(params.input_file_path)

    start = params.slice_start
    end = params.slice_end

   # Read raw data.
    if  (params.full_reconstruction == False) : 
        end = start + 1
    

#    LOG.info('Slice start/end: %s', end)

    proj, flat, dark, theta = dxchange.read_aps_32id(fname, sino=(start, end))
    # LOG.info('Slice start/end: %s, %s', start, end)
    # LOG.info('Data successfully imported: %s', fname)
    # LOG.info('Projections: %s', proj.shape)
    # LOG.info('Flat: %s', flat.shape)
    # LOG.info('Dark: %s', dark.shape)

    # Flat-field correction of raw data.
    data = tomopy.normalize(proj, flat, dark)
    # LOG.info('Normalization completed')

    data = tomopy.downsample(data, level=int(params.binning))
    # LOG.info('Binning: %s', params.binning)

    # remove stripes
    data = tomopy.remove_stripe_fw(data,level=5,wname='sym16',sigma=1,pad=True)
    # LOG.info('Ring removal completed')    

    # phase retrieval
    #data = tomopy.prep.phase.retrieve_phase(data,pixel_size=detector_pixel_size_x,dist=sample_detector_distance,energy=monochromator_energy,alpha=8e-3,pad=True)

    # Find rotation center
    #rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)

    # Set rotation center.
    rot_center = params.center/np.power(2, float(params.binning))
    # LOG.info('Rotation center: %s', rot_center)

    data = tomopy.minus_log(data)
    # LOG.info('Minus log compled')

    # Reconstruct object using Gridrec algorithm.
    # LOG.info('Reconstruction started using %s', params.reconstruction_algorithm)
    if (str(params.reconstruction_algorithm) == 'sirt'):
        # LOG.info('Iteration: %s', params.iteration_count)
        rec = tomopy.recon(data, theta,  center=rot_center, algorithm='sirt', num_iter=params.iteration_count)
    else:
        # LOG.info('Filter: %s', params.filter)
        rec = tomopy.recon(data, theta, center=rot_center, algorithm='gridrec', filter_name=params.filter)

    # LOG.info('Reconstrion of %s completed', rec.shape)

    # Mask each reconstructed slice with a circle.
    rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

    
    if (params.dry_run == False):
         # Write data as stack of TIFs.
        fname = str(params.output_path) + 'reco'
        dxchange.write_tiff_stack(rec, fname=fname, overwrite=True)
        # LOG.info('Reconstrcution saved: %s', fname)

    if  (params.full_reconstruction == False) :
        return rec

