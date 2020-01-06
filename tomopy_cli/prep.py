import tomopy
import numpy as np

from tomopy_cli import log


def data(proj, flat, dark, params):
    # zinger_removal
    proj, flat = zinger_removal(proj, flat, params)

    if (params.dark_zero):
        dark *= 0
    # normalize
    data = flat_correction(proj, flat, dark, params)
    # remove stripes
    data = remove_stripe(data, params)
    # phase retrieval
    data = phase_retrieval(data, params)
    # minus log
    data = minus_log(data, params)
    # remove outlier
    data = remove_nan_neg_inf(data, params)

    return data


def binning(data, params):

    rot_center = params.rotation_axis / np.power(2, float(params.binning))
    if (params.binning == 0):
        log.info("  *** rotation center: %f" % rot_center)
    else:
        log.warning("  *** binning: %d" % params.binning)
        log.warning("  *** rotation center: %f" % rot_center)


    data = tomopy.downsample(data, level=int(params.binning)) 
    data = tomopy.downsample(data, level=int(params.binning), axis=1)

    return data, rot_center

def padding(data, params):

    log.info("  *** padding")
    N = data.shape[2]
    data_pad = np.zeros([data.shape[0],data.shape[1],3*N//2],dtype = "float32")
    data_pad[:,:,N//4:5*N//4] = data
    data_pad[:,:,0:N//4] = np.reshape(data[:,:,0],[data.shape[0],data.shape[1],1])
    data_pad[:,:,5*N//4:] = np.reshape(data[:,:,-1],[data.shape[0],data.shape[1],1])

    data = data_pad
    rot_center = params.rotation_axis + N//4

    return data, rot_center

def remove_nan_neg_inf(data, params):

    log.info('  *** remove nan, neg and inf')
    if(params.fix_nan_and_inf == True):
        log.info('  *** *** standard')
        log.info('  *** *** replacement value %f ' % params.fix_nan_and_inf_value)
        data = tomopy.remove_nan(data, val=params.fix_nan_and_inf_value)
        data = tomopy.remove_neg(data, val=params.fix_nan_and_inf_value)
        data[np.where(data == np.inf)] = params.fix_nan_and_inf_value
    else:
        log.warning('  *** *** none')

    return data

def zinger_removal(proj, flat, params):

    log.info("  *** zinger removal")
    if (params.zinger_removal_method == 'standard'):
        log.info('  *** *** standard')
        log.info("  *** *** zinger level projections: %d" % params.zinger_level_projections)
        log.info("  *** *** zinger level white: %s" % params.zinger_level_white)
        log.info("  *** *** zinger_size: %d" % params.zinger_size)
        proj = tomopy.misc.corr.remove_outlier(proj, params.zinger_level_projections, size=params.zinger_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(flat, params.zinger_level_white, size=params.zinger_size, axis=0)
    elif(params.phase_retrieval_method == 'none'):
        log.warning('  *** *** none')

    return proj, flat


def flat_correction(proj, flat, dark, params):

    log.info('  *** normalization')
    if(params.flat_correction_method == 'standard'):
        data = tomopy.normalize(proj, flat, dark, cutoff=params.normalization_cutoff)
        log.info('  *** *** standard %f cut-off' % params.normalization_cutoff)
    elif(params.flat_correction_method == 'air'):
        data = tomopy.normalize_bg(proj, air=params.air)
        log.info('  *** *** air %d pixels' % params.air)
    elif(params.flat_correction_method == 'none'):
        data = proj
        log.warning('  *** *** normalization is turned off')

    return data

def remove_stripe(data, params):

    log.info('  *** remove stripe:')
    if(params.stripe_removal_method == 'fourier-wavelet'):
        log.info('  *** *** fourier wavelet')
        data = tomopy.remove_stripe_fw(data,level=params.fourier_wavelet_level,wname=params.fourier_wavelet_filter,sigma=params.fourier_wavelet_sigma,pad=params.fourier_wavelet_pad)
        log.info('  *** ***  *** level %d ' % params.fourier_wavelet_level)
        log.info('  *** ***  *** wname %s ' % params.fourier_wavelet_filter)
        log.info('  *** ***  *** sigma %f ' % params.fourier_wavelet_sigma)
        log.info('  *** ***  *** pad %r ' % params.fourier_wavelet_pad)
    elif(params.stripe_removal_method == 'titarenko'):
        log.info('  *** *** titarenko')
        data = tomopy.remove_stripe_ti(data, nblock=params.titarenko_nblock, alpha=params.titarenko_alpha)
        log.info('  *** ***  *** nblock %d ' % params.titarenko_nblock)
        log.info('  *** ***  *** alpha %f ' % params.titarenko_alpha)
    elif(params.stripe_removal_method == 'smoothing-filter'):
        log.info('  *** *** smoothing filter')
        data = tomopy.remove_stripe_sf(data,  size==params.smoothing_filter_size)
        log.info('  *** ***  *** size %d ' % params.smoothing_filter_size)
    elif(params.stripe_removal_method == 'none'):
        log.warning('  *** *** none')

    return data

def phase_retrieval(data, params):
    
    log.info("  *** phase retrieval")
    if (params.phase_retrieval_method == 'paganin'):
        log.info('  *** *** paganin')
        log.info("  *** *** pixel size: %s" % params.pixel_size)
        log.info("  *** *** sample detector distance: %s" % params.propagation_distance)
        log.info("  *** *** energy: %s" % params.energy)
        log.info("  *** *** alpha: %s" % params.alpha)
        data = tomopy.phase.retrieve_phase(data,pixel_size=(params.pixel_size*1e-4),dist=(params.propagation_distance/10.0),energy=params.energy, alpha=params.alpha,pad=True)
    elif(params.phase_retrieval_method == 'none'):
        log.warning('  *** *** none')

    return data
   
def minus_log(data, params):

    log.info("  *** minus log")
    if(params.minus_log):
        log.info('  *** *** ON')
        data = tomopy.minus_log(data)
    else:
        log.warning('  *** *** OFF')

    return data

