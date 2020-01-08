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
from tomopy_cli import proc


def rec(params):
    
    data_shape = file_io.get_dx_dims(params)

    if params.rotation_axis < 0:
        params.rotation_axis =  data_shape[2]/2

    # Select sinogram range to reconstruct
    if (params.reconstruction_type == "full"):
        nSino_per_chunk = params.nsino_per_chunk
        chunks = int(np.ceil(data_shape[1]/nSino_per_chunk))    
        sino_start = 0
        sino_end = chunks*nSino_per_chunk

    else: # "slice" and "try"       
        nSino_per_chunk = pow(2, int(params.binning))
        chunks = 1
        ssino = int(data_shape[1] * params.nsino)
        sino_start = ssino
        sino_end = sino_start + pow(2, int(params.binning)) 

    strt = 0
    for iChunk in range(0, chunks):
        log.info('chunk # %i' % (iChunk))
        sino_chunk_start = np.int(sino_start + nSino_per_chunk*iChunk)
        sino_chunk_end = np.int(sino_start + nSino_per_chunk*(iChunk+1))
        log.info('  *** [%i, %i]' % (sino_chunk_start/pow(2, int(params.binning)), sino_chunk_end/pow(2, int(params.binning))))
                
        if sino_chunk_end > sino_end: 
            break

        sino = (int(sino_chunk_start), int(sino_chunk_end))

        # Read APS 32-BM raw data.
        proj, flat, dark, theta, rotation_axis = file_io.read_tomo(sino, params)

        # apply all preprocessing functions
        data = prep.data(proj, flat, dark, params)

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
            log.warning('  Reconstruct slice [%d] with rotation axis range [%.2f - %.2f] in [%.2f] pixel steps' % (ssino, center_range[0], center_range[1], center_range[2]))

            rotation_axis = np.arange(*center_range)
            rec = proc.padded_rec(stack, theta, rotation_axis, params)

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
            log.warning("Reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % \
                       ((sino_end - sino_start)/pow(2, int(params.binning)), sino_start/pow(2, int(params.binning)), sino_end/pow(2, int(params.binning)), \
                       chunks, nSino_per_chunk/pow(2, int(params.binning))))            

            rec = proc.padded_rec(data, theta, rotation_axis, params)

            # Save images to a temporary folder.
            tail = os.sep + os.path.splitext(os.path.basename(params.hdf_file))[0]+ '_full_rec' + os.sep 
            fname = os.path.dirname(params.hdf_file) + '_rec' + tail + 'recon'
            log_fname = os.path.dirname(params.hdf_file) + '_rec' + tail + os.path.split(params.config)[1]
            if (params.reconstruction_type == "full"):
                if(iChunk == chunks-1):
                    log.info("handling of the last chunk")
                    log.info("  *** chunk # %d" % (chunks))
                    log.info("  *** last rec size %d" % ((data_shape[1]-(chunks-1)*nSino_per_chunk)/pow(2, int(params.binning))))
                    rec = rec[0:data_shape[1]-(chunks-1)*nSino_per_chunk,:,:]
                
            dxchange.write_tiff_stack(rec, fname=fname, start=strt)
            strt += int((sino[1] - sino[0]) / np.power(2, float(params.binning)))
        log.info("  *** reconstructions: %s" % fname)

    # make a copy of the tomopy.conf in the reconstructed data directory path
    # in this way you can reproduce the reconstruction by simply running:
    # $ tomopy recon --config /path/tomopy.conf
    try:
        shutil.copy(params.config, log_fname)
        log.info('  *** copied %s to %s ' % (params.config, log_fname))
    except:
        pass


