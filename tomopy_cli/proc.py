import tomopy
import numpy as np

from tomopy_cli import log


def padded_rec(data, theta, rotation_axis, params):

       # original shape
      N = data.shape[2]
      # padding
      data, padded_rotation_axis = padding(data, rotation_axis) 

      # Reconstruct object
      rec = reconstruct(data, theta, padded_rotation_axis, params)
      # restore shape 
      rec = rec[:,N//4:5*N//4,N//4:5*N//4]
      # Mask each reconstructed slice with a circle
      rec = mask(rec, params)

      return rec


def padding(data, rotation_axis):

    log.info("  *** padding")
    N = data.shape[2]
    data_pad = np.zeros([data.shape[0],data.shape[1],3*N//2],dtype = "float32")
    data_pad[:,:,N//4:5*N//4] = data
    data_pad[:,:,0:N//4] = np.reshape(data[:,:,0],[data.shape[0],data.shape[1],1])
    data_pad[:,:,5*N//4:] = np.reshape(data[:,:,-1],[data.shape[0],data.shape[1],1])

    data = data_pad
    rot_center = rotation_axis + N//4

    return data, rot_center

def reconstruct(data, theta, rot_center, params):

    if(params.reconstruction_type == "try"):
        sinogram_order = True
    else:
        sinogram_order = False
               
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
    elif params.reconstruction_algorithm == 'gridrec':
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.filter)
    else:
        log.warning("  *** *** algorithm: %s is not supported yet" % params.reconstruction_algorithm)
        params.reconstruction_algorithm = 'gridrec'
        log.warning("  *** *** using: %s instead" % params.reconstruction_algorithm)
        log.warning("  *** *** sinogram_order: %s" % sinogram_order)
        rec = tomopy.recon(data, theta, center=rot_center, sinogram_order=sinogram_order, algorithm=params.reconstruction_algorithm, filter_name=params.filter)

    return rec


def mask(data, params):
    
    if(params.reconstruction_mask):
        if 0 < params.reconstruction_mask_ratio <= 1:
            log.warning("  *** apply reconstruction mask ratio: %f " % params.reconstruction_mask_ratio)
            data = tomopy.circ_mask(data, axis=0, ratio=params.reconstruction_mask_ratio)
        else:
            log.error("  *** mask ratio must be between 0-1: %f is ignored" % params.reconstruction_mask_ratio)

    return data