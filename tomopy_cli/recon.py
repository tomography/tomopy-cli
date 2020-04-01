import os
import sys
import shutil
from pathlib import Path
from multiprocessing import cpu_count
import threading
import numpy as np
import tomopy
import dxchange

from tomopy_cli import log
from tomopy_cli import file_io
from tomopy_cli import prep
from tomopy_cli import beamhardening
from tomopy_cli import find_center

def rec(params):
    
    data_shape = file_io.get_dx_dims(params)

    #Read parameters from DXchange file if requested
    params = file_io.auto_read_dxchange(params)
    if params.rotation_axis <= 0:
        params.rotation_axis =  data_shape[2]/2
        log.warning('  *** *** No rotation center given: assuming the middle of the projections at %f' % float(params.rotation_axis))
    
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
        #If params.nsino_per_chunk < 1, use # of processor cores
        if params.nsino_per_chunk < 1:
            params.nsino_per_chunk = cpu_count()
        nSino_per_chunk = params.nsino_per_chunk * pow(2, int(params.binning))
        chunks = int(np.ceil((sino_end - sino_start)/nSino_per_chunk))    
    elif (params.reconstruction_type == 'try'):
        _try_rec(params)
        return
    else: # "slice"        
        nSino_per_chunk = pow(2, int(params.binning))
        chunks = 1
        ssino = int(data_shape[1] * params.nsino)
        sino_start = ssino
        sino_end = sino_start + pow(2, int(params.binning)) 

    log.info("reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % \
               ((sino_end - sino_start)/pow(2, int(params.binning)), sino_start/pow(2, int(params.binning)), sino_end/pow(2, int(params.binning)), \
               chunks, nSino_per_chunk/pow(2, int(params.binning))))            

    strt = sino_start
    for iChunk in range(0, chunks):
        log.info('chunk # %i/%i' % (iChunk + 1, chunks))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        if sino_chunk_end > sino_end:
            log.warning('  *** asking to go to row {0:d}, but our end row is {1:d}'.format(sino_chunk_end, sino_end))
            sino_chunk_end = sino_end
        log.info('  *** [%i, %i]' % (sino_chunk_start/pow(2, int(params.binning)), sino_chunk_end/pow(2, int(params.binning))))
                
        sino = np.array((int(sino_chunk_start), int(sino_chunk_end)))
        
        phase_pad = 8 # params.retrieve_phase_pad #Francesco, please add this parameter to the command line, default 8
        # extra data for padded phase retrieval
        if(params.retrieve_phase_method=="paganin"):
                sino[0] -= (iChunk>0)*phase_pad
                sino[1] += (iChunk<chunks-1)*phase_pad
                log.info('  *** extra padding for phase retrieval gives slices [%i,%i] ' % (sino[0],sino[1]))
        # extra data for padded phase retrieval
        # Read APS 32-BM raw data.
        proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params)
        # What if sino overruns the size of data?
        if sino[1] - sino[0] > proj.shape[1]:
            log.warning(" *** Chunk size > remaining data size.")
            sino = (sino[0], sino[0] + proj.shape[1])

        # apply all preprocessing functions
        data = prep.all(proj, flat, dark, params, sino)

        # unpad after phase retrieval
        if(params.retrieve_phase_method=="paganin"):
                data = data[:,(iChunk>0)*phase_pad:-(iChunk<chunks-1)*phase_pad-(phase_pad==0)]
                sino[0] += (iChunk>0)*phase_pad
                sino[1] -= (iChunk<chunks-1)*phase_pad
                log.info('  *** unpadding after phase retrieval gives slices [%i,%i] ' % (sino[0],sino[1]))
 
        # Reconstruct: this is for "slice" and "full" methods
        rec = padded_rec(data, theta, rotation_axis, params)
        # Save images
        if (params.reconstruction_type == "full"):
            tail = os.sep + os.path.splitext(os.path.basename(params.file_name))[0]+ '_rec' + os.sep 
            fname = os.path.dirname(params.file_name) + '_rec' + tail + 'recon'
            write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                            args = (rec,),
                                            kwargs = {'fname':fname, 'start':strt, 'overwrite':True})
            write_thread.start()
            strt += int((sino[1] - sino[0]) / np.power(2, float(params.binning)))
        if (params.reconstruction_type == "slice"):
            fname = Path.joinpath(Path(os.path.dirname(params.file_name) + '_rec'), 
                                    'slice_rec', 'recon_'+ Path(params.file_name).stem)
            dxchange.write_tiff_stack(rec, fname=str(fname), overwrite=False)

        log.info("  *** reconstructions: %s" % fname)
    

def _try_rec(params):
    log.info("  *** *** starting 'try' reconstruction") 
    data_shape = file_io.get_dx_dims(params)
    # Select sinogram range to reconstruct
    nSino_per_chunk = pow(2, int(params.binning))
    sino_start = int(data_shape[1] * params.nsino)
    sino_end = sino_start + pow(2, int(params.binning))
    if sino_end > data_shape[1]:
        log.warning('  *** *** *** binning would request row past end of data.  Truncating.')
        sino_start = data_shape[1] - pow(2, int(params.binning)) 
        sino_end = data_shape[1]

    log.info("reconstructing a slice binned from raw data rows [%d] to [%d]" % \
               (sino_start, sino_end))

    log.info('  *** binned rows [%i, %i]' % (sino_start/pow(2, int(params.binning)), sino_end/pow(2, int(params.binning))))
            
    sino = (int(sino_start), int(sino_end))

    #Set up the centers of rotation we will use
    # Read APS 32-BM raw data.
    proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params, True)
    # apply all preprocessing functions
    data = prep.all(proj, flat, dark, params, sino)
    rec = []
    center_range = []
    # try passes an array of rotation centers and this is only supported by gridrec
    reconstruction_algorithm_org = params.reconstruction_algorithm
    params.reconstruction_algorithm = 'gridrec'

    if (params.file_type == 'standard'):
        center_search_width = params.center_search_width/np.power(2, float(params.binning))
        center_range = np.arange(rotation_axis-center_search_width, rotation_axis+center_search_width, 0.5)
        stack = np.empty((len(center_range), data_shape[0], int(data_shape[2])))
        for i, axis in enumerate(center_range):
            stack[i] = data[:, 0, :]
        log.warning('  reconstruct slice [%d] with rotation axis range [%.2f - %.2f] in [%.2f] pixel steps' 
                        % (sino_start, center_range[0], center_range[-1], center_range[1] - center_range[0]))

        rec = padded_rec(stack, theta, center_range, params)

    else:
        rotation_axis = params.rotation_axis_flip // pow(2,int(params.binning))
        center_search_width = params.center_search_width/np.power(2, float(params.binning))
        center_range = np.arange(rotation_axis-center_search_width, rotation_axis+center_search_width, 0.5)
        stitched_data = []
        rot_centers = np.zeros_like(center_range)
        #Loop through the assumed rotation centers
        for i, rot_center in enumerate(center_range): 
            params.rotation_axis_flip = rot_center
            stitched_data.append(file_io.flip_and_stitch(params, data, np.ones_like(data[0,...]),
                                                                np.zeros_like(data[0,...]))[0])
            rot_centers[i] = params.rotation_axis
        total_cols = np.min([i.shape[2] for i in stitched_data])
        theta180 = theta[:len(theta)//2] # take first half
        stack = np.empty((len(center_range), theta180.shape[0], total_cols))
        for i in range(center_range.shape[0]):
            stack[i] = stitched_data[i][:theta180.shape[0],0,:total_cols]
        del(stitched_data)
        rec = padded_rec(stack, theta180, rot_centers, params)

    # Save images to a temporary folder.
    fname = (os.path.dirname(params.file_name) + '_rec' + os.sep 
                + 'try_center' + os.sep + file_io.path_base_name(params.file_name) + os.sep + 'recon_')
    for i,axis in enumerate(center_range):
        rfname = fname + str('{0:.2f}'.format(axis*np.power(2, float(params.binning))) + '.tiff')
        dxchange.write_tiff(rec[i], fname=rfname, overwrite=True)
    # restore original method
    params.reconstruction_algorithm = reconstruction_algorithm_org


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

    log.info("  *** padding")
    if((params.reconstruction_algorithm=='gridrec' and params.gridrec_padding)
        or (params.reconstruction_algorithm=='lprec_fbp' and params.lprec_fbp_padding)):
    #if(params.padding):
        log.info('  *** *** ON')
        N = data.shape[2]
        data_pad = np.zeros([data.shape[0],data.shape[1],3*N//2],dtype = "float32")
        data_pad[:,:,N//4:5*N//4] = data
        data_pad[:,:,0:N//4] = np.reshape(data[:,:,0],[data.shape[0],data.shape[1],1])
        data_pad[:,:,5*N//4:] = np.reshape(data[:,:,-1],[data.shape[0],data.shape[1],1])

        data = data_pad
        rot_center = rotation_axis + N//4
    else:
        log.warning('  *** *** OFF')
        data = data
        rot_center = rotation_axis

    return data, rot_center


def unpadding(rec, N, params):

    log.info("  *** un-padding")
    if((params.reconstruction_algorithm=='gridrec' and params.gridrec_padding)
        or (params.reconstruction_algorithm=='lprec_fbp' and params.lprec_fbp_padding)):
    #if(params.padding):
        log.info('  *** *** ON')
        rec = rec[:,N//4:5*N//4,N//4:5*N//4]
    else:
        log.warning('  *** *** OFF')
        rec = rec
    return rec


def reconstruct(data, theta, rot_center, params):

    if(params.reconstruction_type == "try"):
        sinogram_order = True
    else:
        sinogram_order = False
               
    log.info("  *** algorithm: %s" % params.reconstruction_algorithm)
    if params.reconstruction_algorithm == 'astrasirt':
        extra_options ={}
        try:
            extra_options['MinConstraint'] = float(params.astrasirt_min_constraint)
        except ValueError:
            log.warning('Invalid astrasirt_min_constraint value.  Ignoring.')
        try:
            extra_options['MaxConstraint'] = float(params.astrasirt_max_constraint)
        except ValueError:
            log.warning('Invalid astrasirt_max_constraint value.  Ignoring.')
        options = {'proj_type':params.astrasirt_proj_type,
                    'method': params.astrasirt_method,
                    'num_iter': params.astrasirt_num_iter,
                    'extra_options': extra_options,}
        if params.astrasirt_bootstrap:
            log.info('  *** *** bootstrapping with gridrec')
            rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter)
            rec = tomopy.misc.corr.gaussian_filter(rec, axis=1)
            rec = tomopy.misc.corr.gaussian_filter(rec, axis=2)
        shift = (int((data.shape[2]/2 - rot_center)+.5))
        data = np.roll(data, shift, axis=2)
        if params.astrasirt_bootstrap:
            log.info('  *** *** using gridrec to start astrasirt recon')
            rec = tomopy.recon(data, theta, init_recon=rec, algorithm=tomopy.astra, options=options)
        else:
            rec = tomopy.recon(data, theta, algorithm=tomopy.astra, center=params.rotation_axis, options=options)
    elif params.reconstruction_algorithm == 'astrasart':
        extra_options ={}
        try:
            extra_options['MinConstraint'] = float(params.astrasart_min_constraint)
        except ValueError:
            log.warning('Invalid astrasart_min_constraint value.  Ignoring.')
        try:
            extra_options['MaxConstraint'] = float(params.astrasart_max_constraint)
        except ValueError:
            log.warning('Invalid astrasart_max_constraint value.  Ignoring.')
        options = {'proj_type':params.astrasart_proj_type,
                    'method': params.astrasart_method,
                    'num_iter': params.astrasart_num_iter * data.shape[0],
                    'extra_options': extra_options,}
        if params.astrasart_bootstrap:
            log.info('  *** *** bootstrapping with gridrec')
            rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter)
        shift = (int((data.shape[2]/2 - rot_center)+.5))
        data = np.roll(data, shift, axis=2)
        if params.astrasart_bootstrap:
            log.info('  *** *** using gridrec to start astrasart recon')
            rec = tomopy.recon(data, theta, init_recon=rec, algorithm=tomopy.astra, options=options)
        else:
            rec = tomopy.recon(data, theta, algorithm=tomopy.astra, options=options)
        rec = tomopy.recon(data, theta, algorithm=tomopy.astra, options=options)
    elif params.reconstruction_algorithm == 'astracgls':
        extra_options ={}
        options = {'proj_type':params.astracgls_proj_type,
                    'method': params.astracgls_method,
                    'num_iter': params.astracgls_num_iter,
                    'extra_options': extra_options,}
        if params.astracgls_bootstrap:
            log.info('  *** *** bootstrapping with gridrec')
            rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter)
        shift = (int((data.shape[2]/2 - rot_center)+.5))
        data = np.roll(data, shift, axis=2)
        if params.astracgls_bootstrap:
            log.info('  *** *** using gridrec to start astracgls recon')
            rec = tomopy.recon(data, theta, init_recon=rec, algorithm=tomopy.astra, options=options)
        else:
            rec = tomopy.recon(data, theta, algorithm=tomopy.astra, options=options)
    elif params.reconstruction_algorithm == 'gridrec':
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter)
    elif params.reconstruction_algorithm == 'lprec_fbp':
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm=tomopy.lprec,
                            lpmethod='fbp', 
                            filter_name=params.lprec_fbp_filter)
    else:
        log.warning("  *** *** algorithm: %s is not supported yet" % params.reconstruction_algorithm)
        params.reconstruction_algorithm = 'gridrec'
        log.warning("  *** *** using: %s instead" % params.reconstruction_algorithm)
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.filter)
    log.info("  *** reconstruction finished")
    return rec


def mask(data, params):

    log.info("  *** mask")
    if(params.reconstruction_mask):
        log.info('  *** *** ON')
        if 0 < params.reconstruction_mask_ratio <= 1:
            log.warning("  *** mask ratio: %f " % params.reconstruction_mask_ratio)
            data = tomopy.circ_mask(data, axis=0, ratio=params.reconstruction_mask_ratio)
            log.info('  *** masking finished')
        else:
            log.error("  *** mask ratio must be between 0-1: %f is ignored" % params.reconstruction_mask_ratio)
    else:
        log.warning('  *** *** OFF')
    return data
