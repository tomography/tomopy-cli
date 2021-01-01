import sys
import shutil
from pathlib import Path
from multiprocessing import cpu_count
import threading
import logging

import matplotlib.pyplot as plt
import numpy as np
import tomopy
import dxchange

from tomopy_cli import file_io
from tomopy_cli import config
from tomopy_cli import prep
from tomopy_cli import beamhardening
from tomopy_cli import find_center


log = logging.getLogger(__name__)


def rec(params):
    
    data_shape = file_io.get_dx_dims(params)

    # Read parameters from DXchange file if requested
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
        # If params.nsino_per_chunk < 1, use # of processor cores
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

    log.info("  *** reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % (
        (sino_end - sino_start) / pow(2, int(params.binning)),
        sino_start/pow(2, int(params.binning)),
        sino_end/pow(2, int(params.binning)),
        chunks, nSino_per_chunk/pow(2, int(params.binning))))
    
    strt = sino_start
    write_threads = []
    if chunks == 0:
        log.warning("  *** 0 chunks selected for reconstruction, "
                    "check your *start_row*, "
                    "*end_row*, and *nsino_per_chunk*.")
    for iChunk in range(0, chunks):
        log.info('chunk # %i/%i' % (iChunk + 1, chunks))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        if sino_chunk_end > sino_end:
            log.warning('  *** asking to go to row {0:d}, but our end row is {1:d}'.format(sino_chunk_end, sino_end))
            sino_chunk_end = sino_end
        log.info('  *** [%i, %i]' % (sino_chunk_start/pow(2, int(params.binning)), sino_chunk_end/pow(2, int(params.binning))))
        sino = np.array((int(sino_chunk_start), int(sino_chunk_end)))
        # extra data for padded phase retrieval
        if params.retrieve_phase_method == "paganin":
                phase_pad = np.zeros(2,dtype=int)
                if(iChunk>0):
                    phase_pad[0] = -params.retrieve_phase_pad
                if (iChunk<chunks-1):
                    phase_pad[1] =  params.retrieve_phase_pad
                sino += phase_pad
                log.info('  *** extra padding for phase retrieval gives slices [%i,%i] to be read from memory ' % (sino[0],sino[1]))
        # Read APS 32-BM raw data.
        proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params) 
        # What if sino overruns the size of data?
        if sino[1] - sino[0] > proj.shape[1]:
            log.warning("  *** Chunk size > remaining data size.")
            sino = [sino[0], sino[0] + proj.shape[1]]        
        
        # apply all preprocessing functions
        data = prep.all(proj, flat, dark, params, sino)
        # unpad after phase retrieval
        if params.retrieve_phase_method == "paganin":
                phase_pad //= pow(2, int(params.binning))
                sino -= phase_pad                                
                data = data[:,-phase_pad[0]:data.shape[1]-phase_pad[1]]                
                log.info('  *** unpadding after phase retrieval gives slices [%i,%i] ' % (sino[0],sino[1]))
        
        # Reconstruct: this is for "slice" and "full" methods
        rec = padded_rec(data, theta, rotation_axis, params)
        # Save images
        recon_base_dir = reconstruction_folder(params)
        fpath = Path(params.file_name).resolve()
        if params.reconstruction_type == "full":
            recon_dir = recon_base_dir / "{}_rec".format(fpath.stem)
            if params.output_format == 'tiff_stack':
                fname = recon_dir / 'recon'
                log.debug("Full tiff dir: %s", fname)
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args = (rec,),
                                                kwargs = {'fname': str(fname),
                                                          'start': strt,
                                                          'overwrite': True})
            elif params.output_format == "hdf5":
                # HDF5 output
                fname = "{}.hdf".format(recon_dir)
                # file_io.write_hdf5(rec, fname=str(fname), dest_idx=slice(strt, strt+rec.shape[0]),
                #                    maxsize=(sino_end, *rec.shape[1:]), overwrite=(iChunk==0))
                ds_end = int(np.ceil(sino_end / pow(2, int(params.binning))))
                write_thread = threading.Thread(target=file_io.write_hdf5,
                                                args = (rec,),
                                                kwargs = {'fname': str(fname),
                                                          'dest_idx': slice(strt, strt+rec.shape[0]),
                                                          'maxsize': (ds_end, *rec.shape[1:]),
                                                          'overwrite': iChunk==0})
            else:
                log.error("  *** Unknown output_format '%s'", params.output_format)
                fname = "<Not saved (bad output-format)>"
                write_thread = None
            # Save the data to disk
            if write_thread is not None:
                write_thread.start()
                write_threads.append(write_thread)
            # Increment counter for which chunks to save
            strt += (sino[1] - sino[0])
        elif params.reconstruction_type == "slice":
            # Construct the path for where to save the tiffs
            fname = recon_base_dir / 'slice_rec' / 'recon_{}'.format(fpath.stem)
            dxchange.write_tiff(rec, fname=str(fname), overwrite=False)
        else:
            raise ValueError("Unknown value for *reconstruction type*: {}. "
                             "Valid options are {}"
                             "".format(params.reconstruction_type,
                                       config.SECTIONS['reconstruction']['reconstruction-type']['choices']))
        log.info("  *** reconstructions: %s" % fname)
    # Wait until the all threads are done writing data
    for thread in write_threads:
        thread.join()

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

    # Set up the centers of rotation we will use
    # Read APS 32-BM raw data.
    proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params, True)
    # Apply all preprocessing functions
    data = prep.all(proj, flat, dark, params, sino)
    rec = []
    center_range = []
    # try passes an array of rotation centers and this is only supported by gridrec
    # reconstruction_algorithm_org = params.reconstruction_algorithm
    # params.reconstruction_algorithm = 'gridrec'

    if (params.file_type == 'standard'):
        center_search_width = params.center_search_width/np.power(2, float(params.binning))
        center_range = np.arange(rotation_axis-center_search_width, rotation_axis+center_search_width, 0.5)
        # stack = np.empty((len(center_range), data_shape[0], int(data_shape[2])))
        if (params.blocked_views):
            blocked_views = params.blocked_views_end - params.blocked_views_start
            stack = np.empty((len(center_range), data_shape[0]-blocked_views, int(data_shape[2])))
        else:
            stack = np.empty((len(center_range), data.shape[0], int(data.shape[2])))

        for i, axis in enumerate(center_range):
            stack[i] = data[:, 0, :]
        log.warning('  reconstruct slice [%d] with rotation axis range [%.2f - %.2f] in [%.2f] pixel steps' 
                        % (sino_start, center_range[0], center_range[-1], center_range[1] - center_range[0]))
        if params.reconstruction_algorithm == 'gridrec':
            rec = padded_rec(stack, theta, center_range, params)
        else:
            log.warning("  *** Doing try_center with '%s' instead of 'gridrec' is slow.", params.reconstruction_algorithm)
            rec = []
            for center in center_range:
                rec.append(padded_rec(data[:, 0:1, :], theta, center, params))
            rec = np.asarray(rec)
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
    fpath = Path(params.file_name).resolve()
    fbase = Path("{}_rec".format(fpath.parent)) / 'try_center' / fpath.stem
     # fname = fname.resolve().parent / 'slice_rec' / 'recon_{}'.format(fname.stem)

    for i,axis in enumerate(center_range):
        this_center = axis * np.power(2, float(params.binning))
        rfname = fbase / "recon_{:.2f}.tiff".format(this_center)
        dxchange.write_tiff(rec[i], fname=str(rfname), overwrite=True)
    # restore original method
    # params.reconstruction_algorithm = reconstruction_algorithm_org


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
    do_gridrec_padding = params.reconstruction_algorithm=='gridrec' and params.gridrec_padding
    do_lprec_fbp_padding = params.reconstruction_algorithm=='lprec_fbp' and params.lprec_fbp_padding
    if do_gridrec_padding or do_lprec_fbp_padding:
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
    # Check for sane input values
    if not np.all(np.isfinite(data)):
        log.warning("  *** nan/inf found in input data. "
                    "Consider using ``--fix-nan-and-inf True``.")
    log.info("  *** algorithm: %s" % params.reconstruction_algorithm)
    # Apply the various reconstruction algorithms
    if params.reconstruction_algorithm == 'astrasirt':
        extra_options ={}
        try:
            extra_options['MinConstraint'] = float(params.astrasirt_min_constraint)
        except ValueError:
            log.warning("  *** *** invalid astrasirt_min_constraint value..."
                        "ignoring.")
        try:
            extra_options['MaxConstraint'] = float(params.astrasirt_max_constraint)
        except ValueError:
            log.warning("  *** *** invalid astrasirt_max_constraint value..."
                        "ignoring.")
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
        # shift = (int((data.shape[2]/2 - rot_center)+.5))
        # data = np.roll(data, shift, axis=2)
        recon_kw = dict(center=rot_center, algorithm=tomopy.astra,
                        options=options)
        if params.astrasirt_bootstrap:
            log.info('  *** *** using gridrec to start astrasirt recon')
            recon_kw['init_recon'] = rec
        rec = tomopy.recon(data, theta, **recon_kw)
    elif params.reconstruction_algorithm == 'astrasart':
        extra_options ={}
        try:
            extra_options['MinConstraint'] = float(params.astrasart_min_constraint)
        except ValueError:
            log.warning("  *** *** invalid astrasart_min_constraint value..."
                        "ignoring.")
        try:
            extra_options['MaxConstraint'] = float(params.astrasart_max_constraint)
        except ValueError:
            log.warning("  *** *** invalid astrasart_max_constraint value..."
                        "ignoring.")
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
    # gridrec                
    elif params.reconstruction_algorithm == 'gridrec':
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        # import pdb; pdb.set_trace()
        rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter)
    
    # log-polar based method                            
    elif params.reconstruction_algorithm == 'lprec':
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        lpmethod  = params.lprec_method
        
        if (lpmethod=='fbp'):           
            filter_name = params.lprec_fbp_filter 
        else:
            filter_name = 'none'
        rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm=tomopy.lprec,
                            lpmethod=lpmethod,
                            filter_name=filter_name,
                            ncore=1,
                            num_iter=params.lprec_num_iter,
                            reg_par=params.lprec_reg,
                            gpu_list=range(params.lprec_num_gpu))
    else:
        log.warning("  *** *** algorithm: %s is not supported yet" % params.reconstruction_algorithm)
        params.reconstruction_algorithm = 'gridrec'
        log.warning("  *** *** using: %s instead" % params.reconstruction_algorithm)
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.gridrec_filter)
    # Check for sane values
    if np.all(np.isnan(rec)):
        log.error("  *** *** reconstruction produced all NaN")
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


def reconstruction_folder(params):
    """Build the path to the folder that will receive the reconstruction.
    
    """
    file_path = Path(params.file_name).resolve()
    folder_fmt = params.output_folder
    # Format the folder name with the config parameters
    if file_path.is_dir():
        file_name_parent = file_path
    else:
        file_name_parent = file_path.parent
    folder_fmt = folder_fmt.format(file_name_parent=file_name_parent, **params.__dict__)
    return Path(folder_fmt)
