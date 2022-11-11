import os
import logging
import time

import tomopy
import dxchange
import numpy as np

from tomopy_cli import file_io
from tomopy_cli import beamhardening
from tomopy_cli import config

__all__ = ['all', 'remove_nan_neg_inf', 'cap_sinogram_values', 'zinger_removal', 'flat_correction', 
           'remove_stripe', 'phase_retrieval', 'minus_log', 'beamhardening_correct']


log = logging.getLogger(__name__)


def all(proj, flat, dark, params, sino):
    # zinger_removal
    time_start_all = time.time()
    proj, flat = zinger_removal(proj, flat, params)
    if (params.dark_zero):
        dark *= 0
        log.warning('  *** *** dark fields are ignored')

    # normalize
    data = flat_correction(proj, flat, dark, params)
    del(proj, flat, dark)
    # remove stripes
    data = remove_stripe(data, params)
    # Perform beam hardening.  This leaves the data in pathlength.
    if params.beam_hardening_method == 'standard':
        data = beamhardening_correct(data, params, sino)
    else:
        # phase retrieval
        data = phase_retrieval(data, params)
        # minus log
        data = minus_log(data, params)
    # remove outlier
    data = remove_nan_neg_inf(data, params)
    log.debug('  *** total time for prep.all = {0:6.4f}'.format(
                time.time() - time_start_all))
    data = cap_sinogram_values(data, params)
    return data


def remove_nan_neg_inf(data, params):
    time_start_remove_nan = time.time()
    log.info('  *** remove nan, neg and inf')
    if(params.fix_nan_and_inf == True):
        log.info('  *** *** ON')
        log.info('  *** *** replacement value %f ' % params.fix_nan_and_inf_value)
        data = tomopy.remove_nan(data, val=params.fix_nan_and_inf_value)
        data = tomopy.remove_neg(data, val= 0.0)
        data[np.isinf(data)] = params.fix_nan_and_inf_value
    else:
        log.warning('  *** *** OFF')
    log.debug('  *** total time for prep.remove_nan_neg_inf = {0:6.4f}'.format(
                time.time() - time_start_remove_nan))
    return data


def cap_sinogram_values(data, params):
    log.info('  *** cap sinogram max value: %f', params.sinogram_max_value)
    data[data > params.sinogram_max_value] = params.sinogram_max_value
    return data


def zinger_removal(proj, flat, params):

    log.info("  *** zinger removal")
    time_start_zinger = time.time()
    if (params.zinger_removal_method == 'standard'):
        log.info('  *** *** ON')
        log.info("  *** *** zinger level projections: %d" % params.zinger_level_projections)
        log.info("  *** *** zinger level white: %s" % params.zinger_level_white)
        log.info("  *** *** zinger_size: %d" % params.zinger_size)
        proj = tomopy.misc.corr.remove_outlier(proj, params.zinger_level_projections, size=params.zinger_size, axis=0)
        flat = tomopy.misc.corr.remove_outlier(flat, params.zinger_level_white, size=params.zinger_size, axis=0)
    elif(params.zinger_removal_method == 'none'):
        log.warning('  *** *** OFF')
    log.debug('  *** total time for prep.zinger_removal = {0:6.4f}'.format(
                time.time() - time_start_zinger))
    return proj, flat


def flat_correction(proj, flat, dark, params):

    log.info('  *** normalization')
    time_start_flat = time.time()
    if(params.flat_correction_method == 'standard'):
        try:
            data = tomopy.normalize(proj, flat, dark, 
                                cutoff=params.normalization_cutoff / params.bright_exp_ratio)
            data *= params.bright_exp_ratio
        except AttributeError:
            log.warning('  *** *** No bright_exp_ratio found.  Ignore')
        log.info('  *** *** ON %f cut-off' % params.normalization_cutoff)
    elif(params.flat_correction_method == 'air'):
        data = tomopy.normalize_bg(proj, air=params.air)
        log.info('  *** *** air %d pixels' % params.air)
    elif(params.flat_correction_method == 'none'):
        data = proj
        log.warning('  *** *** normalization is turned off')
    else:
        raise ValueError("Unknown value for *flat_correction_method*: {}. "
                         "Valid options are {}"
                         "".format(params.flat_correction_method,
                                   config.SECTIONS['flat-correction']['flat-correction-method']['choices']))
    #Convert 16-bit floats to 32-bit floats
    if data.dtype == np.float16:
        data = data.astype(np.float32, copy=False)
    log.debug('  *** total time for prep.flat_correction = {0:6.4f}'.format(
                time.time() - time_start_flat))
    return data


def remove_stripe(data, params):
    time_start_stripe = time.time()
    log.info('  *** remove stripe:')
    if(params.remove_stripe_method == 'fw'):
        log.info('  *** *** fourier wavelet')
        data = tomopy.remove_stripe_fw(data,level=params.fw_level,wname=params.fw_filter,sigma=params.fw_sigma,pad=params.fw_pad)
        log.info('  *** ***  *** fw level %d ' % params.fw_level)
        log.info('  *** ***  *** fw wname %s ' % params.fw_filter)
        log.info('  *** ***  *** fw sigma %f ' % params.fw_sigma)
        log.info('  *** ***  *** fw pad %r ' % params.fw_pad)
    elif(params.remove_stripe_method == 'ti'):
        log.info('  *** *** titarenko')
        data = tomopy.remove_stripe_ti(data, nblock=params.ti_nblock, alpha=params.ti_alpha)
        log.info('  *** ***  *** ti nblock %d ' % params.ti_nblock)
        log.info('  *** ***  *** ti alpha %f ' % params.ti_alpha)
    elif(params.remove_stripe_method == 'sf'):
        log.info('  *** *** smoothing filter')
        data = tomopy.remove_stripe_sf(data,  size=params.sf_size)
        log.info('  *** ***  *** sf size %d ' % params.sf_size)
    elif(params.remove_stripe_method == 'vo-all'):
        log.info('  *** *** Vo\'s algorithms: all')
        data = tomopy.remove_all_stripe(data, snr=params.vo_all_snr,
                                        la_size=params.vo_all_la_size,
                                        sm_size=params.vo_all_sm_size)
    elif(params.remove_stripe_method == 'none'):
        log.warning('  *** *** OFF')
    log.debug('  *** total time for prep.remove_stripe = {0:6.4f}'.format(
                time.time() - time_start_stripe))

    return data


def phase_retrieval(data, params):
    
    log.info("  *** retrieve phase")
    if (params.retrieve_phase_method == 'paganin'):
        log.info('  *** *** paganin')
        log.info("  *** *** pixel size: %s" % params.pixel_size)
        log.info("  *** *** sample detector distance: %s" % params.propagation_distance)
        log.info("  *** *** energy: %s" % params.energy)
        log.info("  *** *** alpha: %s" % params.retrieve_phase_alpha)
        data = tomopy.retrieve_phase(data,pixel_size=(params.pixel_size*1e-4),dist=(params.propagation_distance/10.0),energy=params.energy, alpha=params.retrieve_phase_alpha,pad=True)
    elif(params.retrieve_phase_method == 'none'):
        log.warning('  *** *** OFF')

    return data
   

def minus_log(data, params):

    log.info("  *** minus log")
    if(params.minus_log):
        log.info('  *** *** ON')
        data = tomopy.minus_log(data)
    else:
        log.warning('  *** *** OFF')

    return data

def beamhardening_correct(data, params, sino):
    """
    Performs beam hardening corrections.
    Inputs
    data: data normalized already for bright and dark corrections.
    params: processing parameters
    sino: row numbers for these data
    """
    log.info("  *** correct beam hardening")
    time_start_bh = time.time()
    data_dtype = data.dtype
    # Correct for centerline of fan
    softener = beamhardening.BeamSoftener(params)
    data = softener.fcorrect_as_pathlength_centerline(data)
    # Make an array of correction factors
    softener.center_row = params.center_row
    log.info("  *** *** Beam hardening center row = {:f}".format(softener.center_row))
    angles = np.abs(np.arange(sino[0], sino[1])- softener.center_row).astype(data_dtype)
    angles *= softener.pixel_size / softener.d_source
    log.info("  *** *** angles from {0:f} to {1:f} urad".format(angles[0], angles[-1]))
    correction_factor = softener.angular_spline(angles).astype(data_dtype)
    if len(data.shape) == 2:
        output = data* correction_factor[:,None]
    else:
        output = data * correction_factor[None, :, None]
    log.debug('  *** total time for prep.remove_stripe = {0:6.4f}'.format(
                time.time() - time_start_bh))
    return output

