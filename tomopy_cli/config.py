import argparse
import sys
import logging
import configparser
from collections import OrderedDict
import numpy as np

from tomopy_cli import log
from tomopy_cli import util

CONFIG_FILE_NAME = "tomopy.conf"
ROTATION_AXIS_FILE_NAME = "rotation_axis.json"

SECTIONS = OrderedDict()

SECTIONS['general'] = {
    'config': {
        'default': CONFIG_FILE_NAME,
        'type': str,
        'help': "File name of configuration",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'}
        }

SECTIONS['find-center'] = {
    'rotation-axis': {
        'default': ROTATION_AXIS_FILE_NAME,
        'type': str,
        'help': "File name of configuration",
        'metavar': 'FILE'},
    'nsino': {
        'default': 0.5,
        'type': float,
        'help': 'Location of the sinogram used to find center (0 top, 1 bottom)'},
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
        'choices': ['standard', 'blocked_views', 'flip_and_stich', 'mosaic']}
        }

SECTIONS['flat-correction'] = {
    'flat-correction-method': {
        'default': 'normal',
        'type': str,
        'help': "Flat correction method",
        'choices': ['normal', 'air']},
    'normalization-cutoff': {
        'default': 1.0,
        'type': float,
        'help': 'Permitted maximum vaue for the normalized data'},
    'fix-nan-and-inf': {
        'default': False,
        'help': "Fix nan and inf",
        'action': 'store_true'},
    'minus-log': {
        'default': False,
        'help': "Minus log",
        'action': 'store_true'},
        }

SECTIONS['retrieve-phase'] = {
    'phase-retrieval-method': {
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
    'alpha': {
        'default': 0.001,
        'type': float,
        'help': "Regularization parameter"},
    'pad': {
        'default': False,
        'help': "When set, extend the size of the sinogram by padding with zeros",
        'action': 'store_true'}
        }

SECTIONS['stripe-removal'] = {
    'stripe-removal-method': {
        'default': 'none',
        'type': str,
        'help': "Stripe removal method",
        'choices': ['none', 'fourier-wavelet', 'titarenko', 'smoothing-filter']},
    'fourier-wavelet-sigma': {
        'default': 2,
        'type': float,
        'help': "Damping parameter in Fourier space"},
    'fourier-wavelet-filter': {
        'default': 'db5',
        'type': str,
        'help': "Type of the fourier-wavelet filter",
        'choices': ['haar', 'db5', 'sym5']},
    'fourier-wavelet-level': {
        'type': util.positive_int,
        'default': 0,
        'help': "Level parameter used by the fourier-wavelet method"},
    'fourier-wavelet-pad': {
        'default': False,
        'help': "When set, extend the size of the sinogram by padding with zeros",
        'action': 'store_true'},
    'titarenko-alpha': {
        'default': 1.5,
        'type': float,
        'help': "Damping factor"},
    'titarenko-nblock': {
        'default': 0,
        'type': util.positive_int,
        'help': "Number of blocks"},
    'smoothing-filter-size': {
        'default': 5,
        'type': util.positive_int,
        'help': "Size of the smoothing filter."}
        }

SECTIONS['reconstruction'] = {
    'binning': {
        'type': str,
        'default': '0',
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': ['0', '1', '2', '3']},
    'filter': {
        'default': 'parzen',
        'type': str,
        'help': "Reconstruction filter",
        'choices': ['none', 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth']},
    'center': {
        'default': 1024.0,
        'type': float,
        'help': "Location of rotation axis"},
    'iteration-count': {
        'default': 10,
        'type': util.positive_int,
        'help': "Maximum number of iterations"},
    'reconstruction-algorithm': {
        'default': 'gridrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['art', 'bart', 'fpb', 'gridrec', 'mlem', 'osem', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad', 'sirt', 'tv', 'grad', 'tikh']}
        }

RECON_PARAMS = ('file-reading', 'flat-correction', 'stripe-removal', 'retrieve-phase', 'reconstruction')
FIND_CENTER_PARAMS = ('file-reading', 'find-center')

# PREPROC_PARAMS = ('flat-correction', 'stripe-removal', 'retrieve-phase')

NICE_NAMES = ('General', 'Find center', 'File reading', 'Flat correction', 'Retrieve phase', 
              'Stripe removal', 'Reconstruction')

def get_config_name():
    """Get the command line --config option."""
    name = ''
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






