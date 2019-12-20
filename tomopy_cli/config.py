import argparse
import sys
import logging
import configparser
from collections import OrderedDict
import tomopy_cli.util as util
import numpy as np

# LOG = logging.getLogger(__name__)
NAME = "test.conf"
SECTIONS = OrderedDict()

SECTIONS['general'] = {
    'config': {
        'default': NAME,
        'type': str,
        'help': "File name of configuration",
        'metavar': 'FILE'},
    'verbose': {
        'default': False,
        'help': 'Verbose output',
        'action': 'store_true'},
    'log': {
        'default': None,
        'type': str,
        'help': "File name of optional log",
        'metavar': 'FILE'}}
 
SECTIONS['file-io'] = {
    'projection-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first projection"},
    'projection-end': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last projection"},
    'projection-number': {
        'type': util.positive_int,
        'default': 0,
        'help': "Number of projections"},
    'projection-min': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first projection"},
    'projection-max': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last projection"},
    'dark-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first dark field"},
    'dark-end': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last dark field"},
    'dark-min': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first dark field"},
    'dark-max': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last dark field"},
    'flat-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first white field"},
    'flat-end': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last white field"},
    'flat-min': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the first white field"},
    'flat-max': {
        'type': util.positive_int,
        'default': 0,
        'help': "Index of the last white field"},
    'slice-start': {
        'type': util.positive_int,
        'default': 0,
        'help': "Start slice to read for reconstruction"},
    'slice-end': {
        'type': util.positive_int,
        'default': 1,
        'help': "End slice to read for reconstruction"},
    'slice-center': {
        'type': util.positive_int,
        'default': 0,
        'help': "Slice used to find the center of rotation"},
    'input-path': {
        'default': '.',
        'type': str,
        'help': "Path of the last used directory",
        'metavar': 'PATH'},
    'input-file-path': {
        'default': '.',
        'type': str,
        'help': "Name of the last file used",
        'metavar': 'PATH'},
    'output-path': {
        'default': '.',
        'type': str,
        'help': "Path to location or format-specified file path "
                "for storing reconstructed slices",
        'metavar': 'PATH'}}

SECTIONS['flat-field-correction'] = {
    'flat-field': {
        'default': False,
        'help': "Enable flat field correction",
        'action': 'store_true'},
    'flat-field-method': {
        'default': 'default',
        'type': str,
        'help': "Flat field correction method",
        'choices': ['default', 'background', 'roi']},
    'cut-off': {
        'default': 1.0,
        'type': float,
        'help': "Permitted maximum vaue for the normalized data"},
    'air': {
        'type': util.positive_int,
        'default': 1,
        'help': "Number of pixels at each boundary to calculate the scaling factor"},
    'roi-tx': {
        'type': str,
        'default': '0',
        'help': "ROI top left x pixel coordinate"},
    'roi-ty': {
        'type': str,
        'default': '0',
        'help': "ROI top left y pixel coordinate"},
    'roi-bx': {
        'type': str,
        'default': '1',
        'help': "ROI bottom right x pixel coordinate"},
    'roi-by': {
        'type': str,
        'default': '1',
        'help': "ROI bottom right y pixel coordinate"},
    'num-flats': {
        'default': 0,
        'type': int,
        'help': "Number of flats for ffc correction."},
    'manual': {
        'default': False,
        'help': "Allow manual entry for proj, dark, white and theta ranges",
        'action': 'store_true'}}

SECTIONS['normalization'] = {
    'nan-and-inf': {
        'default': True,
        'help': "Fix nan and inf"},
    'minus-log': {
        'default': True,
        'help': 'Do minus log'}}

SECTIONS['phase-retrieval'] = {
    'phase-method': {
        'default': 'none',
        'type': str,
        'help': "Phase retrieval correction method",
        'choices': ['none', 'paganin']},
    'energy': {
        'default': None,
        'type': float,
        'help': "X-ray energy [keV]"},
    'propagation-distance': {
        'default': None,
        'type': float,
        'help': "Sample <-> detector distance [m]"},
    'pixel-size': {
        'default': None,
        'type': float,
        'help': "Pixel size [m]"},
    'alpha': {
        'default': 0.001,
        'type': float,
        'help': "Regularization parameter"},
    'pad': {
        'default': True,
        'help': "If True, extend the size of the sinogram by padding with zeros"}}

SECTIONS['ring-removal'] = {
    'ring-removal-method': {
        'default': 'none',
        'type': str,
        'help': "Ring removal method",
        'choices': ['none', 'wavelet', 'titarenko', 'smoothing']},
    'wavelet-sigma': {
        'default': 2,
        'type': float,
        'help': "Damping parameter in Fourier space"},
    'wavelet-filter': {
        'default': 'db5',
        'type': str,
        'help': "Type of the wavelet filter",
        'choices': ['haar', 'db5', 'sym5']},
    'wavelet-level': {
        'type': util.positive_int,
        'default': 0,
        'help': "Level parameter used by the Fourier-Wavelet method"},
    'wavelet-padding': {
        'default': False,
        'help': "If True, extend the size of the sinogram by padding with zeros",
        'action': 'store_true'}}

SECTIONS['reconstruction'] = {
    'binning': {
        'type': str,
        'default': '0',
        'help': "Reconstruction binning factor as power(2, choice)",
        'choices': ['0', '1', '2', '3']},
    'filter': {
        'default': 'none',
        'type': str,
        'help': "Reconstruction filter",
        'choices': ['none', 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth']},
    'center': {
        'default': 1024.0,
        'type': float,
        'help': "Rotation axis position"},
    'dry-run': {
        'default': False,
        'help': "Reconstruct without writing data",
        'action': 'store_true'},
    'full-reconstruction': {
        'default': False,
        'help': "Full or one slice only reconstruction",
        'action': 'store_true'},
    'reconstruction-algorithm': {
        'default': 'gridrec',
        'type': str,
        'help': "Reconstruction algorithm",
        'choices': ['gridrec', 'fbp', 'mlem', 'sirt', 'sirtfbp']},
    'theta-start': {
        'default': 0,
        'type': float,
        'help': "Angle of the first projection in radians"},
    'theta-end': {
        'default': np.pi,
        'type': float,
        'help': "Angle of the last projection in radians"}}

SECTIONS['ir'] = {
    'iteration-count': {
        'default': 10,
        'type': util.positive_int,
        'help': "Maximum number of iterations"}}

SECTIONS['sirt'] = {
    'relaxation-factor': {
        'default': 0.25,
        'type': float,
        'help': "Relaxation factor"}}

SECTIONS['sirtfbp'] = {
    'lambda': {
        'default': 0.1,
        'type': float,
        'help': "lambda (sirtfbp)"},
    'mu': {
        'default': 0.5,
        'type': float,
        'help': "mu (sirtfbp)"}}

SECTIONS['processing'] = {
    'sino-pass': {
        'type': util.positive_int,
        'default': 16,
        'help': 'Number of sinograms to process per pass'},
    'ncore': {
        'default': None,
        'help': "Number of cores that will be assigned to jobs"},
    'nchunk': {
        'default': None,
        'help': "Chunk size for each core"}}

TOMO_PARAMS = ('file-io', 'flat-field-correction', 'normalization', 'phase-retrieval', 'processing', 'ring-removal', 'reconstruction', 'ir', 'sirt', 'sirtfbp')

NICE_NAMES = ('General', 'Input', 'Flat field correction', 'Sinogram generation',
              'General reconstruction', 'Tomographic reconstruction',
              'Filtered backprojection',
              'Direct Fourier Inversion', 'Iterative reconstruction',
              'SIRT', 'SBTV', 'GUI settings', 'Estimation', 'Performance')

def get_config_name():
    """Get the command line --config option."""
    name = NAME
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


def config_to_list(config_name=NAME):
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
                print('1')
                if isinstance(value, list):
                    value = ', '.join(value)
            else:
                value = opts['default'] if opts['default'] is not None else ''
                print('2', value)
            prefix = '# ' if value is '' else ''

            if name != 'config':
                print('3')
                config.set(section, prefix + name, value)


    print('xxxxxxxxxxxxxxxxxxxx', config_file)
    with open(config_file, 'wb') as f:
        print('4')
        config.write(f)


def log_values(args):
    """Log all values set in the args namespace.

    Arguments are grouped according to their section and logged alphabetically
    using the DEBUG log level thus --verbose is required.
    """
    args = args.__dict__

    for section, name in zip(SECTIONS, NICE_NAMES):
        entries = sorted((k for k in args.keys() if k in SECTIONS[section]))

        if entries:
            # LOG.debug(name)
            print(name)

            for entry in entries:
                value = args[entry] if args[entry] is not None else "-"
                # LOG.debug("  {:<16} {}".format(entry, value))
