import os
import sys
import pathlib
import numpy as np
import tomopy
import dxchange

from tomopy_cli import config #, __version__
from tomopy_cli import log
from tomopy_cli import file_io
from tomopy_cli import util


def tomo(params):

    # print(params)
    fname = params.hdf_file
    nsino = float(params.nsino)
    ra_fname = params.rotation_axis_file

    if os.path.isfile(fname):    
        log.info("Reconstructing a single file: %s" % fname)   
        if params.reconstruction_type == "try":            
            try_center(params)
        elif params.reconstruction_type == "slice":
            rec_slice(params)
        elif params.reconstruction_type == "full":
            rec_full(params)
        else:
            log.error("Option: %s is not supported " % params.reconstruction_type)   

    elif os.path.isdir(fname):
        log.info("Reconstructing a folder containing multiple files")   

    else:
        log.error("Directory or File Name does not exist: %s" % fname)

    # update config file
    sections = config.RECON_PARAMS
    config.write(params.config, args=params, sections=sections)


def try_center(params):

    data_shape = file_io.get_dx_dims(params.hdf_file, 'data')

    log.info(data_shape)
    ssino = int(data_shape[1] * params.nsino)

    # Select sinogram range to reconstruct
    start = ssino
    end = start + 1
    sino = (start, end)

    # Read APS 32-BM raw data.
    proj, flat, dark, theta = file_io.read_tomo(params, sino)

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

    stack, N, rot_center = util.padding(stack, params) 

    # Reconstruct the same slice with a range of centers. 
    log.info('  *** reconstruct slice %d with rotation axis ranging from %.2f to %.2f in %.2f pixel steps' % (ssino, center_range[0], center_range[1], center_range[2]))
    rec = tomopy.recon(stack, theta, center=np.arange(*center_range)+N//4, sinogram_order=True, algorithm=params.reconstruction_algorithm, filter_name=params.filter, nchunk=1)
    rec = rec[:,N//4:5*N//4,N//4:5*N//4]
 
    # Mask each reconstructed slice with a circle.
    #rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

    index = 0
    # Save images to a temporary folder.
    # variableDict['rec_dir'] = os.path.dirname(params.hdf_file) + '_rec'
    fname = os.path.dirname(params.hdf_file) + '_rec' + os.sep + 'try_center' + os.sep + file_io.path_base_name(params.hdf_file) + os.sep + 'recon_' ##+ os.path.splitext(os.path.basename(params.hdf_file))[0]    
    for axis in np.arange(*center_range):
        rfname = fname + str('{0:.2f}'.format(axis*np.power(2, float(params.binning))) + '.tiff')
        dxchange.write_tiff(rec[index], fname=rfname, overwrite=True)
        index = index + 1
    log.info("  *** reconstructions at: %s" % fname)


def rec_slice(params):

    log.info("  *** rec_slice")
    data_shape = file_io.get_dx_dims(params.hdf_file, 'data')
    ssino = int(data_shape[1] * params.nsino)

    # Select sinogram range to reconstruct       
    start = ssino
    end = start + pow(2, int(params.binning))
    sino = (start, end)

    rec = reconstruct(params, sino)

    if os.path.dirname(params.hdf_file) is not '':
       fname = os.path.dirname(params.hdf_file) + '_rec' + os.sep + 'slice_rec/recon_' + os.path.splitext(os.path.basename(params.hdf_file))[0]
    else:
       fname = './slice_rec/recon_' + os.path.splitext(os.path.basename(params.hdf_file))[0]
    dxchange.write_tiff_stack(rec, fname=fname, overwrite=False)
    log.info("  *** rec: %s" % fname)
    log.info("  *** slice: %d" % start)


def reconstruct(params, sino):
    # Read APS 32-BM raw data.
    proj, flat, dark, theta = file_io.read_tomo(params, sino)
    # zinger_removal
    proj = tomopy.misc.corr.remove_outlier(proj, params.zinger_level_projections, size=15, axis=0)
    flat = tomopy.misc.corr.remove_outlier(flat, params.zinger_level_white, size=15, axis=0)

    # temporary for 2017-07 val Loon samples
    #dark *= 0

    # normalize
    data = tomopy.normalize(proj, flat, dark)


    # remove stripes
    data = tomopy.remove_stripe_fw(data,level=params.fourier_wavelet_level,wname=params.fourier_wavelet_filter,sigma=params.fourier_wavelet_sigma,pad=params.fourier_wavelet_pad)

    #data = tomopy.remove_stripe_ti(data, 1.5)
    #data = tomopy.remove_stripe_sf(data, size=150)

    # phase retrieval
    if (params.phase_retrieval_method == 'paganin'):
        log.info("  *** phase retrieval is ON")
        log.info("  *** *** pixel size: %s" % params.pixel_size)
        log.info("  *** *** sample detector distance: %s" % params.propagation_distance)
        log.info("  *** *** energy: %s" % params.energy)
        log.info("  *** *** alpha: %s" % params.alpha)
        data = tomopy.prep.phase.retrieve_phase(data,pixel_size=(params.pixel_size*1e-4),dist=(params.propagation_distance/10.0),energy=params.energy, alpha=params.alpha,pad=True)

    log.info("  *** raw data: %s" % params.hdf_file)

    # if (variableDict['phase'] == False):
    #     data = tomopy.minus_log(data)
    data = tomopy.minus_log(data)

    data = tomopy.remove_nan(data, val=0.0)
    data = tomopy.remove_neg(data, val=0.00)
    data[np.where(data == np.inf)] = 0.00

    rot_center = params.rotation_axis / np.power(2, float(params.binning))
    log.info("  *** rotation center: %f" % rot_center)
    data = tomopy.downsample(data, level=int(params.binning)) 
    data = tomopy.downsample(data, level=int(params.binning), axis=1)

    data, N, rot_center = util.padding(data, params) 

    # Reconstruct object.
    log.info("  *** algorithm: %s" % params.reconstruction_algorithm)
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
    else:        
        rec = tomopy.recon(data, theta, center=rot_center, algorithm=params.reconstruction_algorithm, filter_name=params.filter)


    rec = rec[:,N//4:5*N//4,N//4:5*N//4]

    # Mask each reconstructed slice with a circle.
    #rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)
    return rec


def rec_full(params):
    
    data_shape = file_io.get_dx_dims(params.hdf_file, 'data')

    nSino_per_chunk = params.nsino_per_chunk                # number of sinogram chunks to reconstruct
                                                            # always power of 2               
                                                            # number of sinogram chunks to reconstruct
                                                            # only one chunk at the time is reconstructed
                                                            # allowing for limited RAM machines to complete a full reconstruction
                                                            #
                                                            # set this number based on how much memory your computer has
                                                            # if it cannot complete a full size reconstruction lower it

    chunks = int(np.ceil(data_shape[1]/nSino_per_chunk))    

    # Select sinogram range to reconstruct.
    sino_start = 0
    sino_end = chunks*nSino_per_chunk
    
    log.info("Reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % ((sino_end - sino_start), sino_start, sino_end, chunks, nSino_per_chunk))            

    strt = 0
    for iChunk in range(0,1):
    # for iChunk in range(0,chunks):
        log.info('chunk # %i' % (iChunk))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        log.info('  *** [%i, %i]' % (sino_chunk_start, sino_chunk_end))
                
        if sino_chunk_end > sino_end: 
            break

        sino = (int(sino_chunk_start), int(sino_chunk_end))
        # Reconstruct.
        rec = reconstruct(params, sino)
        if os.path.dirname(params.hdf_file) is not '':
            fname = os.path.dirname(params.hdf_file) + '_rec' + os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_full_rec/' + 'recon'
        else:
            fname = '.' + os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_full_rec/' + 'recon'

        log.info("  *** reconstructions: %s" % fname)

        if(iChunk == chunks-1):
            log.info("handling of the last chunk %d " % iChunk)
            log.info("  *** data_shape %d" % (data_shape[1]))
            log.info("  *** chunks # %d" % (chunks))
            log.info("  *** nSino_per_chunk %d" % (nSino_per_chunk))
            log.info("  *** last rec size %d" % (data_shape[1]-(chunks-1)*nSino_per_chunk))
            rec = rec[0:data_shape[1]-(chunks-1)*nSino_per_chunk,:,:]
            
        dxchange.write_tiff_stack(rec, fname=fname, start=strt)
        strt += int((sino[1] - sino[0]) / np.power(2, float(params.binning)))

    rec_log_msg = "\n" + "tomopy recon --rotation-axis " + str(params.rotation_axis) + " --reconstruction-type full " + params.hdf_file
    if (int(params.binning) > 0):
        rec_log_msg = rec_log_msg + " --bin " + params.binning

    if (params.phase_retrieval_method == 'paganin'):
        rec_log_msg = rec_log_msg + \
        " --phase-retrieval-method " + params.params.phase_retrieval_method + \
        " --propagation-distance " + str(params.propagation_distance) + \
        " ----pixel-size " + str(params.pixel_size) + \
        " --energy " + str(params.energy) + \
        " --alpha " + str(params.alpha)

    # log.info('  *** command to repeat the reconstruction: %s' % rec_log_msg)

    p = pathlib.Path(fname)
    lfname = os.path.join(params.logs_home, p.parts[-3] + '.log')
    log.info('  *** command added to %s ' % lfname)
    with open(lfname, "a") as myfile:
        myfile.write(rec_log_msg)

