import os
import sys
import shutil
import pathlib
import numpy as np
import tomopy
import dxchange

from tomopy_cli import config #, __version__
from tomopy_cli import log
from tomopy_cli import file_io
from tomopy_cli import prep
from tomopy_cli import util
from tomopy_cli import proc


def try_center(params):

    data_shape = file_io.get_dx_dims(params.hdf_file, 'data')

    log.info(data_shape)
    ssino = int(data_shape[1] * params.nsino)

    # Select sinogram range to reconstruct
    start = ssino
    end = start + 1
    sino = (start, end)

    # Read APS 32-BM raw data.
    proj, flat, dark, theta = file_io.read_tomo(sino, params)

    # Flat-field correction of raw data.
    data = tomopy.normalize(proj, flat, dark, cutoff=1.4)

    # remove stripes
    data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)

    data = tomopy.minus_log(data)

    data = tomopy.remove_nan(data, val=0.0)
    data = tomopy.remove_neg(data, val=0.00)
    data[np.where(data == np.inf)] = 0.00

    # downsample
    params.rotation_axis = params.rotation_axis/np.power(2, float(params.binning))
    params.center_search_width = params.center_search_width/np.power(2, float(params.binning))

    center_range = (params.rotation_axis-params.center_search_width, params.rotation_axis+params.center_search_width, 0.5)

    data = tomopy.downsample(data, level=int(params.binning))
    data_shape2 = data_shape[2]
    data_shape2 = data_shape2 / np.power(2, float(params.binning))

    stack = np.empty((len(np.arange(*center_range)), data_shape[0], int(data_shape2)))

    index = 0
    for axis in np.arange(*center_range):
        stack[index] = data[:, 0, :]
        index = index + 1

    # original shape
    N = stack.shape[2]

    # padding
    stack, rot_center = prep.padding(stack, params) 

    # Reconstruct the same slice with a range of centers. 
    log.info('  *** reconstruct slice %d with rotation axis ranging from %.2f to %.2f in %.2f pixel steps' % (ssino, center_range[0], center_range[1], center_range[2]))
    rec = tomopy.recon(stack, theta, center=np.arange(*center_range)+N//4, sinogram_order=True, algorithm=params.reconstruction_algorithm, filter_name=params.filter, nchunk=1)
    rec = rec[:,N//4:5*N//4,N//4:5*N//4]
 
    # Mask each reconstructed slice with a circle.
    #rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

    # Save images to a temporary folder.
    fname = os.path.dirname(params.hdf_file) + '_rec' + os.sep + 'try_center' + os.sep + file_io.path_base_name(params.hdf_file) + os.sep + 'recon_' ##+ os.path.splitext(os.path.basename(params.hdf_file))[0]    

    index = 0
    for axis in np.arange(*center_range):
        rfname = fname + str('{0:.2f}'.format(axis*np.power(2, float(params.binning))) + '.tiff')
        dxchange.write_tiff(rec[index], fname=rfname, overwrite=True)
        index = index + 1
    log.info("  *** reconstructions at: %s" % fname)



def rec_full(params):
    
    data_shape = file_io.get_dx_dims(params.hdf_file, 'data')

    # Select sinogram range to reconstruct
    if (params.reconstruction_type == "full"):
        nSino_per_chunk = params.nsino_per_chunk
        chunks = int(np.ceil(data_shape[1]/nSino_per_chunk))    
        sino_start = 0
        sino_end = chunks*nSino_per_chunk
    else:         
        # slice reconstruction
        nSino_per_chunk = 1
        chunks = 1
        ssino = int(data_shape[1] * params.nsino)
        sino_start = ssino
        sino_end = sino_start + pow(2, int(params.binning)) # not sure the binning actually works ...
        # sino = (start, end)
    
    log.info("Reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % ((sino_end - sino_start), sino_start, sino_end, chunks, nSino_per_chunk))            

    strt = 0
    for iChunk in range(0, chunks):
        log.info('chunk # %i' % (iChunk))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        log.info('  *** [%i, %i]' % (sino_chunk_start, sino_chunk_end))
                
        if sino_chunk_end > sino_end: 
            break

        sino = (int(sino_chunk_start), int(sino_chunk_end))

        # Reconstruct
        rec = rec_chunk(sino, params)

        tail = os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_full_rec' + os.sep 
        fname = os.path.dirname(params.hdf_file) + '_rec' + tail + 'recon'
        log_fname = os.path.dirname(params.hdf_file) + '_rec' + tail + os.path.split(params.config)[1]

        log.info("  *** reconstructions: %s" % fname)

        if (params.reconstruction_type == "full"):
            if(iChunk == chunks-1):
                log.info("handling of the last chunk %d " % iChunk)
                log.info("  *** data_shape %d" % (data_shape[1]))
                log.info("  *** chunks # %d" % (chunks))
                log.info("  *** nSino_per_chunk %d" % (nSino_per_chunk))
                log.info("  *** last rec size %d" % (data_shape[1]-(chunks-1)*nSino_per_chunk))
                rec = rec[0:data_shape[1]-(chunks-1)*nSino_per_chunk,:,:]
            
        dxchange.write_tiff_stack(rec, fname=fname, start=strt)
        strt += int((sino[1] - sino[0]) / np.power(2, float(params.binning)))

    # make a copy of the tomopy.conf in the reconstructed data directory path
    # in this way you can reproduce the reconstruction by simply running:
    #
    # tomopy recon --config /path/tomopy.conf    
    #
    try:
        shutil.copy(params.config, log_fname)
        log.info('  *** copied %s to %s ' % (params.config, log_fname))
    except:
        pass


def rec_chunk(sino, params):

    # Read APS 32-BM raw data.
    proj, flat, dark, theta = file_io.read_tomo(sino, params)
    # zinger_removal
    proj, flat = prep.zinger_removal(proj, flat, params)

    if (params.dark_zero):
        dark *= 0
    # normalize
    data = prep.flat_correction(proj, flat, dark, params)
    # remove stripes
    data = prep.remove_stripe(data, params)
    # phase retrieval
    data = prep.phase_retrieval(data, params)
    # minus log
    data = prep.minus_log(data, params)
    # remove outlier
    data = prep.remove_nan_neg_inf(data, params)
    # binning
    data, rotation_center = prep.binning(data, params)
    # original shape
    N = data.shape[2]
    # padding
    data, rot_center = prep.padding(data, params) 
    # Reconstruct object
    rec = proc.reconstruct(data, theta, rot_center, params)
    # restore shape 
    rec = rec[:,N//4:5*N//4,N//4:5*N//4]
    # Mask each reconstructed slice with a circle
    rec = proc.mask(rec, params)

    return rec



