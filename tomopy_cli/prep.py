import tomopy
import numpy as np

from tomopy_cli import log


def data(proj, flat, dark, params):
    # zinger_removal
    proj, flat = zinger_removal(proj, flat, params)

    if (params.dark_zero):
        dark *= 0
        log.warning('  *** *** dark fields are ignored')

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
    elif(params.zinger_removal_method == 'none'):
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
    """This simplier stripe removal function is more concise and API-change proof!

    The the difference is that now params should contain another dictionary
    which contains only the parameters for this stripe removal function.
    """

    print(params.remove_stripe_fw_argments)
    if params.remove_stripe_method in dir(tomopy):
        log.info(' *** remove stripe:')
        # Here the remove_stripe_method MUST be the actual name of the tomopy function
        log.info(' *** *** %s' % params.remove_stripe_method)
        # If we pack all of the parameters for stripe removal into a dictionary,
        # then we can use one parameter for any stripe removal function
        for key in params.remove_stripe_argments:
            log.info(' *** *** *** {} {}'.format(key, params.remove_stripe_argments[key]))
        data = getattr(tomopy, params.remove_stripe_method)(data, **params.remove_stripe_argments)
    else:
       log.warning('  *** *** none')
       # handle the case where the method is wrong
       pass

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

