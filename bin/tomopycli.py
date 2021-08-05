#!/usr/bin/env python

import os
import re
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import yaml

from tomopy_cli import config, __version__
from tomopy_cli import logging
from tomopy_cli import recon
from tomopy_cli import find_center
from tomopy_cli import file_io
from tomopy_cli import post
from tomopy_cli import flat_drift_correction
from tomopy_cli.auto_complete import create_complete_tomopy
from tomopy_cli.logging import log_exception


log = logging.getLogger('tomopy_cli.bin.tomopy')


KNOWN_FORMATS = ['dx', 'aps2bm', 'aps7bm', 'aps32id']


def init(args):
    if not os.path.exists(str(args.config)):
        config.write(args.config)
    else:
        log.error("{0} already exists".format(args.config))    


def run_status(args):
    config.log_values(args)


def run_find_center(args):
    if (str(args.file_format) in KNOWN_FORMATS):
        log.warning('find center start')
        args = find_center.find_rotation_axis(args)
        log.warning('find center end')
        # update tomopy.conf
        sections = config.RECON_PARAMS
        config.update_config(args, is_reconstruction=False)
    else:
        log.error("  *** %s is not a supported file format" % args.file_format)
        exit()

def run_flat_drift_correction(args):
    if (str(args.file_format) in KNOWN_FORMATS):
        log.warning('flat drift correction start')
        args = flat_drift_correction.flat_drift_correction(args)
        log.warning('flat drift correction end')       
    else:
        log.error("  *** %s is not a supported file format" % args.file_format)
        exit()
        
def run_seg(args):

    log.warning('segmentation start')
    post.segment(args)
    log.warning('segmentation end')

    # update tomopy.conf
    sections = config.RECON_PARAMS
    config.write(args.config, args=args, sections=sections)


def run_convert(args):

    log.warning('convert start')
    file_io.convert(args)
    log.warning('convert end')

    
def run_rec(args):

    log.warning('reconstruction start')
    file_path = Path(args.file_name)
    if str(args.file_format) in KNOWN_FORMATS:
        if file_path.suffix == '.yaml':
            # Load the list of files from a given YAML parameters file
            log.info("Reconstructing files listed in: %s" % file_path)
            file_list = file_io.yaml_file_list(file_path)
            failed_files = []
            for idx, this_fname in enumerate(file_list):
                args.file_name = file_path.parent / this_fname
                log.info("Reconstructing next file (%d/%d): %s",
                         idx, len(file_list), args.file_name)
                try:
                    recon.rec(args)
                except Exception as err:
                    # This file failed, but we can keep going and try the rest of the files
                    failed_files.append(this_fname)
                    # Log the exception and stacktrace
                    log.error("  *** reconstruction failed: %s", repr(err))
                    log_exception(log, err, fmt="      %s")
                else:
                    config.update_config(args)
            # Report list of failed files so it's not buried in the log
            if len(failed_files) > 0:
                log.error("Some tomograms could not be reconstructed: %s",
                          ", ".join([str(f) for f in failed_files]))
        elif file_path.is_file():
            log.info("reconstructing a single file: %s" % args.file_name)
            recon.rec(args)
            config.update_config(args)
        elif file_path.is_dir():
            # Add a trailing slash if missing
            top = os.path.join(args.file_name, '')
            h5_file_list = list(filter(lambda x: x.endswith(('.h5', '.hdf', 'hdf5')), os.listdir(top)))
            if (h5_file_list):
                h5_file_list.sort()
                log.info("found: %s" % h5_file_list) 
                index=0
                for fname in h5_file_list:
                    args.file_name = top + fname
                    log.warning("  *** file %d/%d;  %s" % (index, len(h5_file_list), fname))
                    index += 1
                    recon.rec(args)
                    config.update_config(args)
                log.warning('reconstruction end')
            else:
                log.error("directory %s does not contain any file" % args.file_name)
        else:
            log.error("directory or File Name does not exist: %s" % args.file_name)
    else:
        # add here support for other file formats
        log.error("  *** %s is not a supported file format" % args.file_format)
        log.error("supported data formats are: %s, %s, %s, %s" % tuple(KNOWN_FORMATS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', **config.SECTIONS['general']['config'])
    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    tomo_params = config.RECON_PARAMS
    find_center_params = config.RECON_PARAMS
    convert_params = config.CONVERT_PARAMS
    flat_drift_correction_params = config.CORRECT_PARAMS

    cmd_parsers = [
        ('init',        init,            (),                             "Create configuration file"),
        ('recon',       run_rec,         tomo_params,                    "Run tomographic reconstruction"),
        ('status',      run_status,      tomo_params,                    "Show the tomographic reconstruction status"),
        ('segment',     run_seg,         tomo_params,                    "Run segmentation on reconstured data"),
        ('find_center', run_find_center, find_center_params,             "Find rotation axis location for all hdf files in a directory"),
        ('flat_drift_correction', run_flat_drift_correction, flat_drift_correction_params,             "Fix drift of flat field during data acquistion"),
        ('convert',     run_convert,     convert_params,                 "Convert pre-2015 (proj, dark, white) hdf files in a single data exchange h5 file"),        
    ]

    subparsers = parser.add_subparsers(title="Commands", metavar='')

    for cmd, func, sections, text in cmd_parsers:
        cmd_params = config.Params(sections=sections)
        cmd_parser = subparsers.add_parser(cmd, help=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        cmd_parser = cmd_params.add_arguments(cmd_parser)
        cmd_parser.set_defaults(_func=func)

    args = config.parse_known_args(parser, subparser=True)
    # create logger
    logs_home = args.logs_home

    # make sure logs directory exists
    if not os.path.exists(logs_home):
        os.makedirs(logs_home)

    lfname = os.path.join(logs_home, 'tomopy_' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H_%M_%S") + '.log')

    log_level = 'DEBUG' if args.verbose else "INFO"
    logging.setup_custom_logger(lfname, level=log_level)
    log.debug("Started tomopy_cli")
    log.info("Saving log at %s" % lfname)

    try:
        args._func(args)
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()

