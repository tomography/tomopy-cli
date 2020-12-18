import os
import sys
import shutil
from pathlib import Path
import argparse
import configparser
from collections import OrderedDict
import logging
import warnings
import inspect

import h5py
import numpy as np

import tomopy
from tomopy_cli import util
from tomopy_cli import __version__
from tomopy_cli import recon


log = logging.getLogger(__name__)


def default_parameter(func, param):
    """Get the default value for a function parameter.
    
    For a given function *func*, introspect the function and return
    the default value for the function parameter named *param*.
    
    Return
    ======
    default_val
      The default value for the parameter.
    
    Raises
    ======
    RuntimeError
      Raised if the function *func* has no default value for the
      requested parameter *param*.
    
    """
    # Retrieve the function parameter by introspection
    try:
        sig = inspect.signature(func)
        _param = sig.parameters[param]
    except TypeError as e:
        warnings.warn(str(e))
        log.warning(str(e))
        return None
    # Check if a default value exists
    if _param.default is _param.empty:
        # No default is listed in the function, so throw an exception
        msg = ("No default value given for parameter *{}* of callable {}."
               "".format(param, func))
        raise RuntimeError(msg)
    else:
        # Retrieve and return the parameter's default value
        return _param.default


LOGS_HOME = Path.home()/'logs'
CONFIG_FILE_NAME = Path.home()/'tomopy.conf'
ROTATION_AXIS_FILE_NAME = "rotation_axis.json"

SECTIONS = OrderedDict()


SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration file",
        'metavar': 'FILE'},
    'logs-home': {
        'default': LOGS_HOME,
        'type': str,
        'help': "Log file directory",
        'metavar': 'FILE'},
    'rotation-axis-file': {
        'default': ROTATION_AXIS_FILE_NAME,
        'type': str,
        'help': "File name of rataion axis locations",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'config-update': {
        'default': False,
        'help': 'When set, the content of the config file is updated using the current params values',
        'action': 'store_true'},
        }


SECTIONS['find-rotation-axis'] = {
    'center-search-width': {
        'type': float,
        'default': 10.0,
        'help': "+/- center search width (pixel). Search is in 0.5 pixel increments"},
    'rotation-axis': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis"},
    'rotation-axis-auto': {
        'default': 'read_auto',
        'type': str,
        'help': "How to get rotation axis: read from HDF5, auto calculate, read from json file, or take from this file",
        'choices': ['read_auto', 'read_manual', 'manual', 'auto', 'json']},
    'rotation-axis-flip': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis in a 0-360 flip and stich data collection"},
        }

SECTIONS['file-reading'] = {
    'file-name': {
        'default': '.',
        'type': Path,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'file-format': {
        'default': 'dx',
        'type': str,
        'help': "see from https://dxchange.readthedocs.io/en/latest/source/demo.html",
        'choices': ['dx', 'anka', 'australian', 'als', 'elettra', 'esrf', 'aps1id', 'aps2bm', 'aps5bm', 'aps7bm', 'aps8bm', 'aps13bm', 'aps32id', 'petraP05', 'tomcat', 'xradia']},
    'file-type': {
        'default': 'standard',
        'type': str,
        'help': "Input file type",
        'choices': ['standard', 'flip_and_stich', 'mosaic']},
    'nsino': {
        'default': 0.5,
        'type': float,
        'help': 'Location of the sinogram used for slice reconstruction and find axis (0 top, 1 bottom)'},
    'nsino-per-chunk': {     
        'type': int,
        'default': 256,
        'help': "Number of sinograms per chunk. Use larger numbers with computers with larger memory.  Value <= 0 defaults to # of cpus.",},
    'binning': {
        'type': util.positive_int,
        'default': 0,
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': [0, 1, 2, 3]},
    'reverse': {
        'default': False,
        'help': 'When set, the data set was collected in reverse (180-0)',
        'action': 'store_true'},
    'blocked-views': {
        'default': False,
        'help': 'When set, the blocked-views options are used',
        'action': 'store_true'},
    'dark-zero': {
        'default': False,
        'help': 'When set, the the dark field is set to zero',
        'action': 'store_true'},
    'start-row': {
        'default': 0,
        'type': int,
        'help': 'Row on which to start reconstructions'},
    'end-row': {
        'default': -1,
        'type': int,
        'help': 'Row on which to end reconstruction.  Negative values = last row of projection data.'},
    'scintillator-auto': {
        'default': False,
        'help': "When set, read scintillator properties from the HDF file",
        'action': 'store_true'},
    'pixel-size-auto': {
        'default': False,
        'help': "When set, read effective pixel size from the HDF file",
        'action': 'store_true'},
       }

SECTIONS['dx-options'] = {
    'dx-update': {
        'default': False,
        'help': 'When set, the content of the hdf dx file /process tag is updated using the current params values',
        'action': 'store_true'},
        }

SECTIONS['blocked-views'] = {
    'blocked-views-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Projection number of the first blocked view"},
    'blocked-views-end': {
        'type': util.positive_int,
        'default': 1,
        'help': "Projection number of the first blocked view"},
        }

SECTIONS['zinger-removal'] = {
    'zinger-removal-method': {
        'default': 'none',
        'type': str,
        'help': "Zinger removal correction method",
        'choices': ['none', 'standard']},
    'zinger-level-projections': {
        'default': 800.0,
        'type': float,
        'help': 'Expected difference value between outlier value and the median value of the array'},
    'zinger-level-white': {
        'default': 1000.0,
        'type': float,
        'help': 'Expected difference value between outlier value and the median value of the array'},
    'zinger-size': {
        'type': util.positive_int,
        'default': 3,
        'help': "Size of the median filter"},
        }

SECTIONS['flat-correction'] = {
    'flat-correction-method': {
        'default': 'standard',
        'type': str,
        'help': "Flat correction method",
        'choices': ['standard', 'air', 'none']},
    'normalization-cutoff': {
        'default': 1.0,
        'type': float,
        'help': 'Permitted maximum vaue for the normalized data'},
    'air': {
        'type': util.positive_int,
        'default': 10,
        'help': "Number of pixels at each boundary to calculate the scaling factor"},
    'fix-nan-and-inf': {
        'default': False,
        'help': "Fix nan and inf",
        'action': 'store_true'},
    'fix-nan-and-inf-value': {
        'default': 6.0,
        'type': float,
        'help': "Values to be replaced with negative values in array"},
    'minus-log': {
        'default': True,
        'help': "Minus log",
        'action': 'store_true'},
        }

SECTIONS['retrieve-phase'] = {
    'retrieve-phase-method': {
        'default': 'none',
        'type': str,
        'help': "Phase retrieval correction method",
        'choices': ['none', 'paganin']},
    'energy': {
        'default': 20,
        'type': float,
        'help': "X-ray energy [keV]"},
    'propagation-distance': {
        'default': 60,
        'type': float,
        'help': "Sample detector distance [mm]"},
    'pixel-size': {
        'default': 1.17,
        'type': float,
        'help': "Pixel size [microns]"},
    'retrieve-phase-alpha': {
        'default': 0.001,
        'type': float,
        'help': "Regularization parameter"},
    'retrieve-phase-alpha-try': {
        'default': False,
        'help': "When set, multiple reconstruction of the same slice with different alpha coefficient are generated",
        'action': 'store_true'},
    'retrieve-phase-pad': {
        'type': util.positive_int,
        'default': 8,
        'help': "Padding with extra slices in z for phase-retrieval filtering"},
        }

SECTIONS['remove-stripe'] = {
    'remove-stripe-method': {
        'default': 'none',
        'type': str,
        'help': "Remove stripe method: none, fourier-wavelet, titarenko, smoothing filter, all Vo's algorithms",
        'choices': ['none', 'fw', 'ti', 'sf', 'vo-all']},
        }

SECTIONS['vo-all'] = {
    'vo-all-snr': {
        'default': default_parameter(tomopy.remove_all_stripe, 'snr'),
        'type': float,
        'help': "Ratio used to locate large stripes. Greater is less sensitive."},
    'vo-all-la-size': {
        'default': default_parameter(tomopy.remove_all_stripe, 'la_size'),
        'type': util.positive_int,
        'help': "Window size of the median filter to remove large stripes."},
    'vo-all-sm-size': {
        'default': default_parameter(tomopy.remove_all_stripe, 'sm_size'),
        'type': util.positive_int,
        'help': "Window size of the median filter to remove small-to-medium stripes."},
}

SECTIONS['fw'] = {
    'fw-sigma': {
        'default': 1,
        'type': float,
        'help': "Fourier-Wavelet remove stripe damping parameter"},
    'fw-filter': {
        'default': 'sym16',
        'type': str,
        'help': "Fourier-Wavelet remove stripe filter",
        'choices': ['haar', 'db5', 'sym5', 'sym16']},
    'fw-level': {
        'type': util.positive_int,
        'default': 7,
        'help': "Fourier-Wavelet remove stripe level parameter"},
    'fw-pad': {
        'default': True,
        'help': "When set, Fourier-Wavelet remove stripe extend the size of the sinogram by padding with zeros",
        'action': 'store_true'},
    }

SECTIONS['ti'] = {
    'ti-alpha': {
        'default': default_parameter(tomopy.remove_stripe_ti, 'alpha'),
        'type': float,
        'help': "Titarenko remove stripe damping factor"},
    'ti-nblock': {
        'default': default_parameter(tomopy.remove_stripe_ti, 'nblock'),
        'type': util.positive_int,
        'help': "Titarenko remove stripe number of blocks"},
    }

SECTIONS['sf'] = {
    'sf-size': {
        'default': default_parameter(tomopy.remove_stripe_sf, 'size'),
        'type': util.positive_int,
        'help': "Smoothing filter remove stripe size"}
        }

SECTIONS['beam-hardening']= {
    'beam-hardening-method': {
        'default': 'none',
        'type': str,
        'help': "Beam hardening method.",
        'choices':['none','standard']},
    'source-distance': {
        'default': 36.0,
        'type': float,
        'help': 'Distance from source to scintillator in m'},
    'scintillator-material': {
        'default': 'LuAG_Ce',
        'type': str,
        'help': 'Scintillator material for beam hardening',
        'choices': ['LuAG_Ce', 'LYSO_Ce', 'YAG_Ce']},
    'scintillator-thickness': {
        'default': 100.0,
        'type': float,
        'help': 'Scintillator thickness for beam hardening'},
    'center-row': {
        'default': 0.0,
        'type': float,
        'help': 'Row with the center of the vertical fan for beam hardening.'},
    'sample-material': {
        'default': 'Fe',
        'type': str,
        'help': 'Sample material for beam hardening',
        'choices': ['Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-1-auto': {
        'default': False,
        'help': 'If True, read filter 1 from HDF meta data',},
    'filter-1-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 1 material for beam hardening',
        'choices': ['auto','none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-1-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 1 thickness for beam hardening'},
    'filter-2-auto': {
        'default': False,
        'help': 'If True, read filter 2 from HDF meta data',},
    'filter-2-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 2 material for beam hardening',
        'choices': ['auto','none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-2-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 2 thickness for beam hardening'},
    'filter-3-auto': {
        'default': False,
        'help': 'If True, read filter 3 from HDF meta data',},
    'filter-3-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 3 material for beam hardening',
        'choices': ['none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-3-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 3 thickness for beam hardening'},
    }

SECTIONS['reconstruction'] = {
    'reconstruction-type': {
        'default': 'try',
        'type': str,
        'help': "Reconstruct slice or full data set. For  option (try): multiple reconstruction of the same slice with different (rotation axis) are generated",
        'choices': ['try', 'slice', 'full']},
    'reconstruction-algorithm': {
        'default': 'gridrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['art', 'astrasart','astrasirt', 'astracgls', 'bart', 'fpb', 'gridrec', 'lprec_fbp', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh']},
    'reconstruction-mask': {
        'default': False,
        'help': "When set, applies circular mask to the reconstructed slices",
        'action': 'store_true'},
    'reconstruction-mask-ratio': {
        'default': 1.0,
        'type': float,
        'help': "Ratio of the mask’s diameter in pixels to the smallest edge size along given axis"},
    'output-format': {
        'default': 'tiff_stack',
        'type': str,
        'help': "How to save the reconstructed data. Only applies when ``reconstruction-type == 'full'``.",
        'choices': ['tiff_stack', 'hdf5'],
        },
    'output-folder': {
        'default': "{file_name_parent}_rec",
        'type': str,
        'help': ("Where to save the reconstructed data. Can accept other parameters "
                 "and extra tokens (file_name_parent). "
                 "Eg: \"{file_name_parent}_rec/{reconstruction_algorithm}/\"")
        },
    }

SECTIONS['gridrec'] = {
    'gridrec-filter': {
        'default': 'parzen',
        'type': str,
        'help': 'Filter used for gridrec reconstruction',
        'choices': ['none', 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth']},
    'gridrec-padding': {
        'default': False,
        'help': "When set, raw data are padded/unpadded before/after reconstruction",
        'action': 'store_true'},
    }

SECTIONS['lprec-fbp'] = {
    'lprec-fbp-filter': {
        'default': 'parzen',
        'type': str,
        'help': 'Filter used for lprec-fbp reconstruction',
        'choices': ['none', 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth']},
    'lprec-fbp-padding': {
        'default': False,
        'help': "When set, raw data are padded/unpadded before/after reconstruction",
        'action': 'store_true'},
    }
    
SECTIONS['astrasirt'] = {
    'astrasirt-proj-type': {
        'default': 'cuda',
        'choices': ['cuda', 'linear'],
        'type': str,
        'help': 'Projection type for ASTRA-SIRT.  CPU = linear, GPU = cuda.'},
    'astrasirt-method': {
        'default': 'SIRT_CUDA',
        'type': str,
        'help': 'Parameter passed to ASTRA for ASTRA-SIRT algorithm.'},
    'astrasirt-min-constraint': {
        'default': 'None',
        'type': str,
        'help': 'Minimum constraint for ASTRA-SIRT reconstruction.  None = no constraint.'},
    'astrasirt-max-constraint': {
        'default': 'None',
        'type': str,
        'help': 'Maximum constraint for ASTRA-SIRT reconstruction.  None = no constraint.'},
    'astrasirt-num_iter': {
        'default': 200,
        'type': util.positive_int,
        'help': 'Number of requested iterations for ASTRA-SIRT.'},
    'astrasirt-bootstrap': {
        'default': False,
        'help': 'When set, gridrec is run first and used to initialize ASTRA-SIRT.',
        'action': 'store_true',},
    }

SECTIONS['astrasart'] = {
    'astrasart-proj-type': {
        'default': 'cuda',
        'choices': ['cuda', 'linear'],
        'type': str,
        'help': 'Projection type for ASTRA-SART.  CPU = linear, GPU = cuda.'},
    'astrasart-method': {
        'default': 'SART_CUDA',
        'type': str,
        'help': 'Parameter passed to ASTRA for ASTRA-SART algorithm.'},
    'astrasart-min-constraint': {
        'default': 'None',
        'type': str,
        'help': 'Minimum constraint for ASTRA-SART reconstruction.  None = no constraint.'},
    'astrasart-max-constraint': {
        'default': 'None',
        'type': str,
        'help': 'Maximum constraint for ASTRA-SART reconstruction.  None = no constraint.'},
    'astrasart-num_iter': {
        'default': 200,
        'type': util.positive_int,
        'help': 'Number of requested iterations for ASTRA-SART per projection angle.'},
    'astrasart-bootstrap': {
        'default': False,
        'help': 'When set, gridrec is run first and used to initialize ASTRA-SART.',
        'action': 'store_true',},
    }

SECTIONS['astracgls'] = {
    'astracgls-proj-type': {
        'default': 'cuda',
        'choices': ['cuda', 'linear'],
        'type': str,
        'help': 'Projection type for ASTRA-CGLS.  CPU = linear, GPU = cuda.'},
    'astracgls-method': {
        'default': 'CGLS_CUDA',
        'type': str,
        'help': 'Parameter passed to ASTRA for ASTRA-CGLS algorithm.'},
    'astracgls-num_iter': {
        'default': 200,
        'type': util.positive_int,
        'help': 'Number of requested iterations for ASTRA-CGLS.'},
    'astracgls-bootstrap': {
        'default': False,
        'help': 'When set, gridrec is run first and used to initialize ASTRA-GCLS.',
        'action': 'store_true',},
    }

SECTIONS['convert'] = {
    'old-projection-file-name': {
        'default': '.',
        'type': str,
        'help': "Name of the hdf file containing the projections",
        'metavar': 'PATH'},
    'old-dark-file-name': {
        'default': '.',
        'type': str,
        'help': "Name of the hdf file containing the dark images",
        'metavar': 'PATH'},
    'old-white-file-name': {
        'default': '.',
        'type': str,
        'help': "Name of the hdf file containing the white images",
        'metavar': 'PATH'},
        }

RECON_PARAMS = ('find-rotation-axis', 'file-reading', 'dx-options', 'blocked-views', 'zinger-removal', 'flat-correction', 'remove-stripe', 'vo-all', 'fw', 
                'ti', 'sf', 'retrieve-phase', 'beam-hardening', 'reconstruction', 
                'gridrec', 'lprec-fbp', 'astrasart', 'astrasirt', 'astracgls')
FIND_CENTER_PARAMS = ('file-reading', 'find-rotation-axis', 'dx-options')

CONVERT_PARAMS = ('convert', )
# PREPROC_PARAMS = ('flat-correction', 'remove-stripe', 'retrieve-phase')

NICE_NAMES = ('General', 'Find rotation axis', 'File reading', 'dx-options', 'Missing angles', 'Zinger removal', 'Flat correction', 'Retrieve phase', 
              'Remove stripe','Fourier wavelet', 'Titarenko', 'Smoothing filter', 'Beam hardening', 'Reconstruction', 
                'Gridrec', 'LPRec FBP', 'ASTRA SART (GPU)', 'ASTRA SIRT (GPU)', 'ASTRA CGLS (GPU)', 'Convert')

def get_config_name():
    """Get the command line --config option."""
    name = CONFIG_FILE_NAME
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--config'):
            if arg == '--config':
                return sys.argv[i + 1]
            else:
                name = sys.argv[i].split('--config')[1]
                if name[0] == '=':
                    name = name[1:]
                return name
    return name


def parse_known_args(parser, subparser=False):
    """
    Parse arguments from file and then override by the ones specified on the
    command line. Use *parser* for parsing and is *subparser* is True take into
    account that there is a value on the command line specifying the subparser.
    """
    if len(sys.argv) > 1:
        subparser_value = [sys.argv[1]] if subparser else []
        config_values = config_to_list(config_name=get_config_name())
        values = subparser_value + config_values + sys.argv[1:]
    else:
        raise TypeError("A command is required. See ``tomopy --help`` for detailed usage.")
    return parser.parse_known_args(values)[0]


def config_to_list(config_name=CONFIG_FILE_NAME):
    """
    Read arguments from config file and convert them to a list of keys and
    values as sys.argv does when they are specified on the command line.
    *config_name* is the file name of the config file.
    """
    result = []
    config = configparser.ConfigParser()

    if not config.read([config_name]):
        return []

    for section in SECTIONS:
        for name, opts in ((n, o) for n, o in SECTIONS[section].items() if config.has_option(section, n)):
            value = config.get(section, name)

            if value != '' and value != 'None':
                action = opts.get('action', None)

                if action == 'store_true' and value == 'True':
                    # Only the key is on the command line for this action
                    result.append('--{}'.format(name))

                if not action == 'store_true':
                    if opts.get('nargs', None) == '+':
                        result.append('--{}'.format(name))
                        result.extend((v.strip() for v in value.split(',')))
                    else:
                        result.append('--{}={}'.format(name, value))

    return result


def param_from_dxchange(hdf_file, data_path, attr=None, scalar=True, char_array=False):
    """
    Reads a parameter from the HDF file.
    Inputs
    hdf_file: string path or pathlib.Path object for the HDF file.
    data_path: path to the requested data in the HDF file.
    attr: name of the attribute if this is stored as an attribute (default: None)
    scalar: True if the value is a single valued dataset (dafault: True)
    char_array: if True, interpret as a character array.  Useful for EPICS strings (default: False)
    """
    if not os.path.isfile(hdf_file):
        return None
    with h5py.File(hdf_file,'r') as f:
        try:
            if attr:
                return f[data_path].attrs[attr].decode('ASCII')
            elif char_array:
                return ''.join([chr(i) for i in f[data_path][0]]).strip(chr(0))
            elif scalar:
                return f[data_path][0]
            else:
                return None
        except KeyError:
            return None
    

class Params(object):
    def __init__(self, sections=()):
        self.sections = sections + ('general', )

    def add_parser_args(self, parser):
        for section in self.sections:
            for name in sorted(SECTIONS[section]):
                opts = SECTIONS[section][name]
                parser.add_argument('--{}'.format(name), **opts)

    def add_arguments(self, parser):
        self.add_parser_args(parser)
        return parser

    def get_defaults(self):
        parser = argparse.ArgumentParser()
        self.add_arguments(parser)

        return parser.parse_args('')


def write(config_file, args=None, sections=None):
    """
    Write *config_file* with values from *args* if they are specified,
    otherwise use the defaults. If *sections* are specified, write values from
    *args* only to those sections, use the defaults on the remaining ones.
    """
    config = configparser.ConfigParser()
    for section in SECTIONS:
        config.add_section(section)
        for name, opts in SECTIONS[section].items():
            if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                value = getattr(args, name.replace('-', '_'))
                if isinstance(value, list):
                    # print(type(value), value)
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''

            prefix = '# ' if value == '' else ''

            if name != 'config':
                config.set(section, prefix + name, str(value))

    with open(config_file, 'w') as f:
        config.write(f)


def write_hdf(args=None, sections=None):
    """
    Write in the hdf raw data file the content of *config_file* with values from *args* 
    if they are specified, otherwise use the defaults. If *sections* are specified, 
    write values from *args* only to those sections, use the defaults on the remaining ones.
    """
    if (args == None):
        log.warning("  *** Not saving log data to the HDF file.")

    else:
        with h5py.File(args.file_name,'r+') as hdf_file:
            #If the group we will write to already exists, remove it
            if hdf_file.get('/process/tomopy-cli-' + __version__):
                del(hdf_file['/process/tomopy-cli-' + __version__])
            #dt = h5py.string_dtype(encoding='ascii')
            log.info("  *** tomopy.conf parameter written to /process%s in file %s " % (__version__, args.file_name))
            config = configparser.ConfigParser()
            for section in SECTIONS:
                config.add_section(section)
                for name, opts in SECTIONS[section].items():
                    if args and sections and section in sections and hasattr(args, name.replace('-', '_')):
                        value = getattr(args, name.replace('-', '_'))
                        if isinstance(value, list):
                            # print(type(value), value)
                            value = ', '.join(value)
                    else:
                        value = opts['default'] if opts['default'] is not None else ''

                    prefix = '# ' if value == '' else ''

                    if name != 'config':
                        dataset = '/process' + '/tomopy-cli-' + __version__ + '/' + section + '/'+ name
                        dset_length = len(str(value)) * 2 if len(str(value)) > 5 else 10
                        dt = 'S{0:d}'.format(dset_length)
                        hdf_file.require_dataset(dataset, shape=(1,), dtype=dt)
                        log.info(name + ': ' + str(value))
                        try:
                            hdf_file[dataset][0] = np.string_(str(value))
                        except TypeError:
                            log.error("Could not convert value {}".format(value))
                            raise


def log_values(args):
    """Log all values set in the args namespace.

    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    log.warning('tomopy-cli status start')
    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k.replace('_', '-') in SECTIONS[section]))

        # print('log_values', section, name, entries)
        if entries:
            log.info(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                if (value == 'none'):
                    log.warning("  {:<16} {}".format(entry, value))
                elif (value is not False):
                    log.info("  {:<16} {}".format(entry, value))
                elif (value is False):
                    log.warning("  {:<16} {}".format(entry, value))

    log.warning('tomopy-cli status end')


def update_config(args):
    """Update the corresponding configuration file.
    
    If *args.config_update* is true, the original configuration file
    is updated.
    
    If *args.reconstruction_type* is "full", then a new configuration
    file is created alongside the reconstructed data, with a path
    determined by whether *args.output_format* is "tiff_stack" or
    "hdf5".
    
    """
    sections = RECON_PARAMS
    config_file = Path(args.config).resolve()
    data_file = Path(args.file_name).resolve()
    # write(args.config, args=args, sections=sections)
    if (args.config_update):
        # update tomopy.conf
        write(config_file, args=args, sections=sections)
    if (args.reconstruction_type == "full"):
        recon_dir = recon.reconstruction_folder(args)
        if args.output_format == "hdf5":
            log_fname = recon_dir / "{}_rec_{}".format(data_file.stem, config_file.name)
        else:
            log_fname = recon_dir / "{}_rec".format(data_file.stem) / config_file.name
        try:
            write(log_fname, args=args, sections=sections)
        except Exception as e:
            log.error('  *** attempt to save config to %s failed' % log_fname)
            log.error('  *** *** %s' % e)
        else:
            log.info('  *** saved config to %s ' % (log_fname))
            rerun_msg = ' *** command to repeat the reconstruction: tomopy recon --config {}'
            rerun_msg = rerun_msg.format(log_fname)
            log.warning(rerun_msg)
    if(args.dx_update):
        write_hdf(args, sections)       
