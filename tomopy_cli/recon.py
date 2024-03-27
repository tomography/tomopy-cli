import sys
import shutil
from pathlib import Path
from multiprocessing import cpu_count
import threading
import logging

import numpy as np
import tomopy
import dxchange
import h5py
import meta

from tomopy_cli import file_io
from tomopy_cli import config
from tomopy_cli import prep
from tomopy_cli import find_center

__all__ = ['rec', 'double_fov', 'double_fov_try', 'padded_rec', 'padding', 
           'unpadding', 'reconstruct', 'mask', 'reconstruction_folder'] 

log = logging.getLogger(__name__)


def rec(params):
    
    data_shape = file_io.get_dx_dims(params)
    # Read parameters from YAML file
    try:
        params = config.yaml_args(params, params.parameter_file, str(params.file_name))
    except KeyError:
        pass
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
    
    if(params.start_proj):
        sproj = params.start_proj
    else:    
        sproj = 0
    if(params.end_proj or params.end_proj<0):
        eproj = params.end_proj
    else:    
        eproj = data_shape[0]        
    pproj = (sproj, eproj)        

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
        sino = _compute_sino(iChunk, sino_start, sino_end, nSino_per_chunk, chunks, params) 

        # Read APS 32-BM raw data.
        proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, pproj, params) 
        # What if sino overruns the size of data?
        if sino[1] - sino[0] > proj.shape[1]:
            log.warning("  *** Chunk size > remaining data size.")
            sino = [sino[0], sino[0] + proj.shape[1]]

        # Apply all preprocessing functions
        data = prep.all(proj, flat, dark, params, sino)
        del(proj, flat, dark)

        # unpad after phase retrieval
        if params.retrieve_phase_method == "paganin":
                params.phase_pad //= pow(2, int(params.binning))
                sino -= params.phase_pad                                
                data = data[:,-params.phase_pad[0]:data.shape[1]-params.phase_pad[1]]                
                log.info('  *** unpadding after phase retrieval gives slices [%i,%i] ' % (sino[0],sino[1]))
        
        # Reconstruct: this is for "slice" and "full" methods
        rotation_axis_rec = rotation_axis
        if (params.file_type == 'double_fov'):                                
            if(rotation_axis<data.shape[-1]//2):
                #if rotation center is on the left side of the ROI
                data = data[:,:,::-1]
                rotation_axis_rec = data.shape[-1]-rotation_axis                               
            # double FOV by adding zeros
            data = double_fov(data,rotation_axis_rec)    

        #Perform actual reconstruction
        rec = padded_rec(data, theta, rotation_axis_rec, params)

        # Save images
        recon_base_dir = reconstruction_folder(params)
        fpath = Path(params.file_name).resolve()
        if params.reconstruction_type == "full":
            recon_dir = recon_base_dir / "{}_rec".format(fpath.stem)
            if params.save_format == 'tiff':
                fname = recon_dir / 'recon'
                log.debug("Full tiff dir: %s", fname)
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args = (rec,),
                                                kwargs = {'fname': str(fname),
                                                          'start': strt,
                                                          'overwrite': True})
            elif params.save_format == "h5":
                # HDF5 output
                fname = "{}.hdf".format(recon_dir)
                # file_io.write_hdf5(rec, fname=str(fname), dest_idx=slice(strt, strt+rec.shape[0]),
                #                    maxsize=(sino_end, *rec.shape[1:]), overwrite=(iChunk==0))
                ds_end = int(np.ceil(sino_end / pow(2, int(params.binning))))
                write_thread = threading.Thread(target=file_io.write_hdf5,
                                                args = (rec,),
                                                kwargs = {'fname': str(fname),
                                                          'dname': '/exchange/data',
                                                          'dest_idx': slice(strt, strt+rec.shape[0]),
                                                          'maxsize': (ds_end, *rec.shape[1:]),
                                                          'overwrite': iChunk==0})
            else:
                log.error("  *** Unknown save_format '%s'", params.save_format)
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

    if params.save_format == "h5":
        log.info('adding meta data from the raw to the recon hdf file')
        log.info("  *** raw hdf: %s" % params.file_name)
        log.info("  *** rec hdf: %s" % fname)
        tree, meta_dict = meta.read_hdf(params.file_name)

        with h5py.File(fname, 'a') as hf:
            for key, value in meta_dict.items():
                # print(key, value)
                dset = hf.create_dataset(key, data=value[0])
                if value[1] is not None:
                    dset.attrs['units'] = value[1]

def _compute_sino(iChunk, sino_start, sino_end, nSino_per_chunk, chunks, params):
    '''Computes a 2-element array to give starting and ending slices 
    for this chunk.
    '''
    sino_chunk_start = int(sino_start + nSino_per_chunk*iChunk)
    sino_chunk_end = int(sino_start + nSino_per_chunk*(iChunk+1))
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
        params.phase_pad = phase_pad
    return sino


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

    if(params.start_proj):
        sproj = params.start_proj
    else:    
        sproj = 0
    if(params.end_proj):
        eproj = params.end_proj
    if not params.end_proj or params.end_proj == -1:
        eproj = data_shape[0] + 1        
    pproj = (sproj, eproj)        

    # Set up the centers of rotation we will use
    # Read APS 32-BM raw data.    
    proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, pproj, params, True)
    # Apply all preprocessing functions
    data = prep.all(proj, flat, dark, params, sino)
    rec = []
    center_range = []
    # try passes an array of rotation centers and this is only supported by gridrec
    # reconstruction_algorithm_org = params.reconstruction_algorithm
    # params.reconstruction_algorithm = 'gridrec'
    if (params.file_type == 'standard' or params.file_type == 'double_fov'):
        center_search_width = params.center_search_width/np.power(2, float(params.binning))
        center_range = np.arange(rotation_axis-center_search_width, rotation_axis+center_search_width, 0.5)
        # stack = np.empty((len(center_range), data_shape[0], int(data_shape[2])))
        if (params.blocked_views):
            # blocked_views = params.blocked_views_end - params.blocked_views_start
            # stack = np.empty((len(center_range), data_shape[0]-blocked_views, int(data_shape[2])))
            st = params.blocked_views_start
            end = params.blocked_views_end
            #log.warning('%f %f',st,end)
            ids = np.where(((theta-st)%np.pi<0) + ((theta-st)%np.pi>end-st))[0]
            stack = np.empty((len(center_range), len(ids), int(data_shape[2])))
        else:
            stack = np.empty((len(center_range), data.shape[0], int(data.shape[2])))
        for i, axis in enumerate(center_range):
            stack[i] = data[:, 0, :]
        log.warning('  reconstruct slice [%d] with rotation axis range [%.2f - %.2f] in [%.2f] pixel steps' 
                        % (sino_start, center_range[0], center_range[-1], center_range[1] - center_range[0]))
        center_range_rec = center_range
        if (params.file_type == 'double_fov'):                                
            if(rotation_axis<stack.shape[-1]//2):
                #if rotation center is on the left side of the ROI
                stack = stack[:,:,::-1]
                center_range_rec = stack.shape[-1]-center_range
            # double FOV by adding zeros
            stack = double_fov_try(stack,center_range_rec)                
        if params.reconstruction_algorithm == 'gridrec':
            rec = padded_rec(stack, theta, center_range_rec, params)
        else:
            log.warning("  *** Doing try_center with '%s' instead of 'gridrec' is slow.", params.reconstruction_algorithm)
            rec = []
            for center in center_range_rec:
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
            temp = file_io.flip_and_stitch(params, data, np.ones_like(data[0,...]),
                                                                np.zeros_like(data[0,...]), theta)
            stitched_data.append(temp[0])
            theta180 = temp[3]
            rot_centers[i] = params.rotation_axis
        total_cols = np.min([i.shape[2] for i in stitched_data])
        stack = np.empty((len(center_range), theta180.shape[0], total_cols))
        for i in range(center_range.shape[0]):
            stack[i] = stitched_data[i][:theta180.shape[0],0,:total_cols]
        del(stitched_data)
        rec = padded_rec(stack, theta180, rot_centers, params)
    # Save images to a temporary folder.
    fpath = Path(params.file_name).resolve()
    rec_dir = reconstruction_folder(params) / 'try_center' / fpath.stem
    for i,axis in enumerate(center_range):
        this_center = axis * np.power(2, float(params.binning))
        rfname = rec_dir / "recon_{:.2f}.tiff".format(this_center)
        dxchange.write_tiff(rec[i], fname=str(rfname), overwrite=True)

def double_fov(data,rotation_axis):
    # smooth the sinogram border with a smooth weigting function from 0 to 1
    w = max(1,int(2*(data.shape[-1]-rotation_axis)))    
    v = np.linspace(1,0,w,endpoint=False)
    v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)     
    data[:,:,-w:] *= v    
    # double sinogram size with adding 0
    data = np.pad(data,((0,0),(0,0),(0,data.shape[-1])),'constant')    
    return data

def double_fov_try(data,rotation_axis):
    # smooth the sinogram border with a smooth weigting function from 0 to 1
    for r_axis in range(len(rotation_axis)):        
        w = max(1,int(2*(data.shape[-1]-rotation_axis[r_axis])))    
        v = np.linspace(1,0,w,endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)     
        data[r_axis,:,-w:] *= v
    # double sinogram size with adding 0
    data = np.pad(data,((0,0),(0,0),(0,data.shape[-1])),'constant')    
    return data

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
    do_lprec_padding = params.reconstruction_algorithm=='lprec' and params.lprec_padding
    if do_gridrec_padding or do_lprec_padding:
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
    do_gridrec_padding = params.reconstruction_algorithm=='gridrec' and params.gridrec_padding
    do_lprec_padding = params.reconstruction_algorithm=='lprec' and params.lprec_padding

    if do_gridrec_padding or do_lprec_padding:
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
        if(params.reconstruction_type == "try"):
            # each chunk works with 1 rotation center
            nchunk = 1 
        else:
            nchunk = None
        rec = tomopy.recon(data, theta, 
                            center=rot_center, 
                            sinogram_order=sinogram_order, 
                            algorithm='gridrec', 
                            filter_name=params.gridrec_filter,
                            nchunk = nchunk)
    
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
    folder_fmt = params.save_folder
    # Format the folder name with the config parameters
    if file_path.is_dir():
        file_name_parent = file_path
    else:
        file_name_parent = file_path.parent
    folder_fmt = folder_fmt.format(file_name_parent=file_name_parent, **params.__dict__)
    return Path(folder_fmt)
