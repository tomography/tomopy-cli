import os
import logging
import glob
import tempfile
import sys
import numpy as np
import tomopy
import dxchange

from tomopy_cli import config #, __version__
from tomopy_cli import log


def auto(params):

    # print(params)

    # update config file
    sections = config.FIND_CENTER_PARAMS
    config.write(params.config, args=params, sections=sections)

    return

