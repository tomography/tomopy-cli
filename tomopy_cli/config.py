import os
import sys
import pathlib
import argparse
import configparser
import h5py
import numpy as np

from collections import OrderedDict

from tomopy_cli import log
from tomopy_cli import util
from tomopy_cli import __version__

LOGS_HOME = os.path.join(str(pathlib.Path.home()), 'logs')
CONFIG_FILE_NAME = os.path.join(str(pathlib.Path.home()), 'tomopy.conf')
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
        'action': 'store_true'}
        }

SECTIONS['find-rotation-axis'] = {
    'center-search-width': {
        'type': float,
        'default': 10.0,
        'help': "+/- center search width (pixel). Search is in 0.5 pixel increments"},
        }

SECTIONS['file-reading'] = {
    'hdf-file': {
        'default': '.',
        'type': str,
        'help': "Name of the last used hdf file or directory containing multiple hdf files",
        'metavar': 'PATH'},
    'hdf-file-type': {
        'default': 'standard',
        'type': str,
        'help': "Input file type",
        'choices': ['standard', 'flip_and_stich', 'mosaic']},
    'hdf-file-save-log': {
        'default': False,
        'help': 'When set, the content of the config file is saved in the raw hdf dataset',
        'action': 'store_true'},
    'nsino': {
        'default': 0.5,
        'type': float,
        'help': 'Location of the sinogram used for slice reconstruction and find axis (0 top, 1 bottom)'},
    'nsino-per-chunk': {     
        'type': util.positive_int,
        'default': 32,
        'help': "Number of sinagram per chunk. Use larger numbers with computers with larger memory",
        'choices': [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
    'binning': {
        'type': util.positive_int,
        'default': 0,
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': [0, 1, 2, 3]},
    'rotation-axis': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis"},
    'rotation-axis-flip': {
        'default': -1.0,
        'type': float,
        'help': "Location of rotation axis in a 0-360 flip and stich data collection"},
    'reverse': {
        'default': False,
        'help': 'When set, the data set was collected in reverse (180-0)',
        'action': 'store_true'},
    'blocked-views': {
        'default': False,
        'help': 'When set, the missing-angles options are used',
        'action': 'store_true'},
    'dark-zero': {
        'default': False,
        'help': 'When set, the the dark field is set to zero',
        'action': 'store_true'}        
       }

SECTIONS['missing-angles'] = {
    'missing-angles-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Projection number of the first blocked view"},
    'missing-angles-end': {
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
        'default': 0.0,
        'type': float,
        'help': "Values to be replaced with negative values in array"},
    'minus-log': {
        'default': False,
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
        }

SECTIONS['remove-stripe'] = {
    'remove-stripe-method': {
        'default': 'none',
        'type': str,
        'help': "Remove stripe method: none, fourier-wavelet, titarenko, smoothing filter",
        'choices': ['none', 'fw', 'ti', 'sf']},
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
        'default': False,
        'help': "When set, Fourier-Wavelet remove stripe extend the size of the sinogram by padding with zeros",
        'action': 'store_true'},
    }

SECTIONS['ti'] = {
    'ti-alpha': {
        'default': 1.5,
        'type': float,
        'help': "Titarenko remove stripe damping factor"},
    'ti-nblock': {
        'default': 0,
        'type': util.positive_int,
        'help': "Titarenko remove stripe number of blocks"},
    }

SECTIONS['sf'] = {
    'sf-size': {
        'default': 5,
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
    'filter-1-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 1 material for beam hardening',
        'choices': ['none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-1-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 1 thickness for beam hardening'},
    'filter-2-material': {
        'default': 'none',
        'type': str,
        'help': 'Filter 2 material for beam hardening',
        'choices': ['none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-2-thickness': {
        'default': 0.0,
        'type': float,
        'help': 'Filter 2 thickness for beam hardening'},
    'filter-3-material': {
        'default': 'Be',
        'type': str,
        'help': 'Filter 3 material for beam hardening',
        'choices': ['none','Al','Be','Cu','Fe','Ge','Inconel625','LuAG_Ce','LYSO_Ce','Mo','Pb','Si','SS316','Ta','Ti_6_4','W','YAG_Ce']},
    'filter-3-thickness': {
        'default': 750.0,
        'type': float,
        'help': 'Filter 3 thickness for beam hardening'},
    }

SECTIONS['reconstruction'] = {
    'filter': {
        'default': 'parzen',
        'type': str,
        'help': "Reconstruction filter",
        'choices': ['none', 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth']},
    'reconstruction-type': {
        'default': 'try',
        'type': str,
        'help': "Reconstruct slice or full data set. For  option (try): multiple reconstruction of the same slice with different (rotation axis) are generated",
        'choices': ['try', 'slice', 'full']},
    'reconstruction-algorithm': {
        'default': 'gridrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['art', 'astrasirt', 'astracgls', 'bart', 'fpb', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh']},
    'reconstruction-mask': {
        'default': False,
        'help': "When set, applies circular mask to the reconstructed slices",
        'action': 'store_true'},
    'reconstruction-mask-ratio': {
        'default': 1.0,
        'type': float,
        'help': "Ratio of the mask’s diameter in pixels to the smallest edge size along given axis"},
    'padding': {
        'default': False,
        'help': "When set, raw data are padded/unpadded before/after reconstruction",
        'action': 'store_true'},
        }

SECTIONS['iterative'] = {
    'iteration-count': {
        'default': 10,
        'type': util.positive_int,
        'help': "Maximum number of iterations"},
    }

RECON_PARAMS = ('find-rotation-axis', 'file-reading', 'missing-angles', 'zinger-removal', 'flat-correction', 'remove-stripe', 'fw', 
                'ti', 'sf', 'retrieve-phase', 'beam-hardening', 'reconstruction', 'iterative')
FIND_CENTER_PARAMS = ('file-reading', 'find-rotation-axis')

# PREPROC_PARAMS = ('flat-correction', 'remove-stripe', 'retrieve-phase')

NICE_NAMES = ('General', 'Find rotation axis', 'File reading', 'Missing angles', 'Zinger removal', 'Flat correction', 'Retrieve phase', 
              'Remove stripe','Fourier wavelet', 'Titarenko', 'Smoothing filter', 'Beam hardening', 'Reconstruction', 'Iterative')

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
        #print(subparser_value, config_values, values)
    else:
        values = ""

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

            if value is not '' and value != 'None':
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


def param_from_dxchange(hdf_file, data_path, attr = None, scalar = True, char_array=False):
    """
    Reads a parameter from a DXchange file.
    Inputs
    hdf_file: string path or pathlib.Path object for the DXchange file.
    data_path: path to the requested data in the DXchange file.
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

            prefix = '# ' if value is '' else ''

            if name != 'config':
                config.set(section, prefix + name, str(value))
    with open(config_file, 'w') as f:
        config.write(f)

    if args.hdf_file_save_log:
        write_hdf(config_file, args, sections)


def write_hdf(config_file, args=None, sections=None):
    """
    Write in the hdf raw data file the content of *config_file* with values from *args* 
    if they are specified, otherwise use the defaults. If *sections* are specified, 
    write values from *args* only to those sections, use the defaults on the remaining ones.
    """
    with h5py.File(args.hdf_file,'r+') as hdf_file:
        dt = h5py.string_dtype(encoding='ascii')
        log.info("  *** tomopy.conf parameter written to /process in file %s " % args.hdf_file)
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

                prefix = '# ' if value is '' else ''

                if name != 'config':
                    dataset = '/process' + '/tomopy-cli-' + __version__ + '/' + section + '/'+ name
                    hdf_file.require_dataset(dataset, shape=(1,), dtype=dt)
                    hdf_file[dataset][0] = str(value)


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
                log.info("  {:<16} {}".format(entry, value))

    log.warning('tomopy-cli status end')






