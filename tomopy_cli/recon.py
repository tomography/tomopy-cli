import os
import sys
import shutil
import pathlib
import numpy as np
import tomopy
import dxchange

from tomopy_cli import file_io
from tomopy_cli import prep
from tomopy_cli import beamhardening

import logging
from tomopy_cli import log

# logger = log.setup_logger(__name__)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('test.txt')


def rec(params):
    
    data_shape = file_io.get_dx_dims(params)

    #Determine rotation axis
    if params.rotation_axis < 0:
        if file_io.read_rot_center(params):
            params.rotation_axis = file_io.read_rot_center(params)
        else:
            logger.warning('No rotation center given: assuming the middle of the projections.')
            params.rotation_axis =  data_shape[2]/2

    # If we are performing beam hardening, initialize it
    if params.beam_hardening_method == 'standard':
        beamhardening.parse_params(params)
        center_row = beamhardening.find_center_row(params)
        logger.info("Center row for beam hardening = {0:f}".format(center_row))
        if int(params.binning) > 0:
            center_row /= pow(2, int(params.binning))
            logger.info("Center row after binning = {:f}".format(center_row))
        params.center_row = center_row

    # Select sinogram range to reconstruct
    if (params.reconstruction_type == "full"):
        if params.start_row:
            sino_start = params.start_row
        else:
            sino_start = 0
        if params.end_row < 0:
            sino_end = data_shape[1]
        else:
            sino_end = params.end_row 
        nSino_per_chunk = params.nsino_per_chunk
        chunks = int(np.ceil((sino_end - sino_start)/nSino_per_chunk))    

    else: # "slice" and "try"       
        nSino_per_chunk = pow(2, int(params.binning))
        chunks = 1
        ssino = int(data_shape[1] * params.nsino)
        sino_start = ssino
        sino_end = sino_start + pow(2, int(params.binning)) 

    logger.info("reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % \
               ((sino_end - sino_start)/pow(2, int(params.binning)), sino_start/pow(2, int(params.binning)), sino_end/pow(2, int(params.binning)), \
               chunks, nSino_per_chunk/pow(2, int(params.binning))))            

    strt = sino_start
    for iChunk in range(0, chunks):
        logger.info('chunk # %i/%i' % (iChunk + 1, chunks))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        if sino_chunk_end > sino_end:
            logger.warning('  *** asking to go to row {0:d}, but our end row is {1:d}'.format(sino_chunk_end, sino_end))
            sino_chunk_end = sino_end
        logger.info('  *** [%i, %i]' % (sino_chunk_start/pow(2, int(params.binning)), sino_chunk_end/pow(2, int(params.binning))))
                
        sino = (int(sino_chunk_start), int(sino_chunk_end))

        # Read APS 32-BM raw data.
        proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params)

        # What if sino overruns the size of data?
        if sino[1] - sino[0] > proj.shape[1]:
            logger.warning(" *** Chunk size > remaining data size.")
            sino = (sino[0], sino[0] + proj.shape[1])

        # apply all preprocessing functions
        data = prep.all(proj, flat, dark, params, sino)

        # Reconstruct
        if (params.reconstruction_type == "try"):
            # try passes an array of rotation centers and this is only supported by gridrec
            reconstruction_algorithm_org = params.reconstruction_algorithm
            params.reconstruction_algorithm = 'gridrec'

            center_search_width = params.center_search_width/np.power(2, float(params.binning))
            center_range = (rotation_axis-center_search_width, rotation_axis+center_search_width, 0.5)
            stack = np.empty((len(np.arange(*center_range)), data_shape[0], int(data_shape[2]/ np.power(2, float(params.binning)))))
            index = 0
            for axis in np.arange(*center_range):
                stack[index] = data[:, 0, :]
                index = index + 1
            logger.warning('  reconstruct slice [%d] with rotation axis range [%.2f - %.2f] in [%.2f] pixel steps' % (ssino, center_range[0], center_range[1], center_range[2]))

            rotation_axis = np.arange(*center_range)
            rec = padded_rec(stack, theta, rotation_axis, params)

            # Save images to a temporary folder.
            fname = os.path.dirname(params.hdf_file) + '_rec' + os.sep + 'try_center' + os.sep + file_io.path_base_name(params.hdf_file) + os.sep + 'recon_'
            index = 0
            for axis in np.arange(*center_range):
                rfname = fname + str('{0:.2f}'.format(axis*np.power(2, float(params.binning))) + '.tiff')
                dxchange.write_tiff(rec[index], fname=rfname, overwrite=True)
                index = index + 1

            # restore original method
            params.reconstruction_algorithm = reconstruction_algorithm_org

        else: # "slice" and "full"
            rec = padded_rec(data, theta, rotation_axis, params)
            '''
            # handling of the last chunk 
            if (params.reconstruction_type == "full"):
                if(iChunk == chunks-1):
                    logger.info("handling of the last chunk")
                    logger.info("  *** chunk # %d" % (chunks))
                    logger.info("  *** last rec size %d" % ((data_shape[1]-(chunks-1)*nSino_per_chunk)/pow(2, int(params.binning))))
                    rec = rec[0:data_shape[1]-(chunks-1)*nSino_per_chunk,:,:]
            '''
            # Save images
            if (params.reconstruction_type == "full"):
                tail = os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_rec' + os.sep 
                fname = os.path.dirname(params.hdf_file) + '_rec' + tail + 'recon'
                dxchange.write_tiff_stack(rec, fname=fname, start=strt)
                strt += int((sino[1] - sino[0]) / np.power(2, float(params.binning)))
            if (params.reconstruction_type == "slice"):
                fname = os.path.dirname(params.hdf_file)  + os.sep + 'slice_rec/recon_' + os.path.splitext(os.path.basename(params.hdf_file))[0]
                dxchange.write_tiff_stack(rec, fname=fname, overwrite=False)


        logger.info("  *** reconstructions: %s" % fname)

    

def padded_rec(data, theta, rotation_axis, params):

    # original shape
    N = data.shape[2]
    # padding
    data, padded_rotation_axis = padding(data, rotation_axis, params) 
    # reconstruct object
    rec = reconstruct(data, theta, padded_rotation_axis, params)
    # un-padding - restore shape 
    rec = unpadding(rec, N, params)
    # mask each reconstructed slice with a circle
    rec = mask(rec, params)

    return rec


def padding(data, rotation_axis, params):

    logger.info("  *** padding")

    if(params.padding):
        logger.info('  *** *** ON')
        N = data.shape[2]
        data_pad = np.zeros([data.shape[0],data.shape[1],3*N//2],dtype = "float32")
        data_pad[:,:,N//4:5*N//4] = data
        data_pad[:,:,0:N//4] = np.reshape(data[:,:,0],[data.shape[0],data.shape[1],1])
        data_pad[:,:,5*N//4:] = np.reshape(data[:,:,-1],[data.shape[0],data.shape[1],1])

        data = data_pad
        rot_center = rotation_axis + N//4
    else:
        logger.warning('  *** *** OFF')
        data = data
        rot_center = rotation_axis

    return data, rot_center


def unpadding(rec, N, params):

    logger.info("  *** un-padding")
    if(params.padding):
        logger.info('  *** *** ON')
        rec = rec[:,N//4:5*N//4,N//4:5*N//4]
    else:
        logger.warning('  *** *** OFF')
        rec = rec
    return rec


def reconstruct(data, theta, rot_center, params):

    if(params.reconstruction_type == "try"):
        sinogram_order = True
    else:
        sinogram_order = False
               
    logger.info("  *** algorithm: %s" % params.reconstruction_algorithm)
    if params.reconstruction_algorithm == 'astrasirt':
        extra_options ={'MinConstraint':0}
        options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
        shift = (int((data.shape[2]/2 - rot_center)+.5))
        data = np.roll(data, shift, axis=2)
        rec = tomopy.recon(data, theta, algorithm=tomopy.astra, options=options)
    elif params.reconstruction_algorithm == 'astracgls':
        extra_options ={'MinConstraint':0}
        options = {'proj_type':'cuda', 'method':'CGLS_CUDA', 'num_iter':15, 'extra_options':extra_options}
        shift = (int((data.shape[2]/2 - rot_center)+.5))
        data = np.roll(data, shift, axis=2)
        rec = tomopy.recon(data, theta, algorithm=tomopy.astra, options=options)
    elif params.reconstruction_algorithm == 'gridrec':
        logger.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.filter)
    else:
        logger.warning("  *** *** algorithm: %s is not supported yet" % params.reconstruction_algorithm)
        params.reconstruction_algorithm = 'gridrec'
        logger.warning("  *** *** using: %s instead" % params.reconstruction_algorithm)
        logger.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.filter)

    return rec


def mask(data, params):

    logger.info("  *** mask")
    if(params.reconstruction_mask):
        logger.info('  *** *** ON')
        if 0 < params.reconstruction_mask_ratio <= 1:
            logger.warning("  *** mask ratio: %f " % params.reconstruction_mask_ratio)
            data = tomopy.circ_mask(data, axis=0, ratio=params.reconstruction_mask_ratio)
        else:
            logger.error("  *** mask ratio must be between 0-1: %f is ignored" % params.reconstruction_mask_ratio)
    else:
        logger.warning('  *** *** OFF')
    return data

