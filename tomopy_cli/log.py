'''Customized logging for the tomopy_cli library.

Logging in tomopy_cli is built upon the standard logging functionality
in python. This module provides a ``getLogger`` module that can be
used to get a logger object with the usual *debug*, *info*, etc.,
methods. If ``setup_custom_logger`` is called, all subsequent calls to
``getLogger`` will produce a logger that reflects these
customizations.

'''
import logging
import warnings


logger = logging.getLogger(__name__)


def old_logging(func):
    def inside(*args, **kwargs):
        warnings.warn(
            "``tomopy_cli.log.{}`` is deprecated, use the python logging module directly."
            "".format(str(func)[10:17]),
            DeprecationWarning,
        )
        return func(*args, **kwargs)
    return inside


@old_logging
def debug(msg):
    logger.debug(msg)


@old_logging
def info(msg):
    logger.info(msg)


@old_logging
def error(msg):
    logger.error(msg)


@old_logging
def warning(msg):
    logger.warning(msg)


def setup_custom_logger(lfname: str=None, stream_to_console: bool=True, level=logging.DEBUG):
    """Prepare the logging system with custom formatting.
    
    This adds handlers to the *tomopy_cli* parent logger. Any logger
    inside tomopy_cli will produce output based on this functions
    customization parameters.
    
    Parameters
    ----------
    lfname
      Path to where the log file should be stored. If omitted, no file
      logging will be performed.
    stream_to_console
      If true, logs will be output to the console with color
      formatting.
    level
      A logging level for the handler. This can be either a string
      ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), or an actual
      level defined in the python logging framework.
    
    """
    parent_name = __name__.split('.')[0]  # Nominally "tomopy_cli"
    parent_logger = logging.getLogger(parent_name)
    parent_logger.setLevel(level)
    # Set up normal output to a file
    if lfname is not None:
        fHandler = logging.FileHandler(lfname)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s(%(lineno)s) - %(levelname)s: %(message)s')
        fHandler.setFormatter(file_formatter)
        parent_logger.addHandler(fHandler)
    # Set up formatted output to the console
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredLogFormatter('%(asctime)s - %(message)s'))
        parent_logger.addHandler(ch)


class ColoredLogFormatter(logging.Formatter):
    """A logging formatter that add console color codes."""
    __BLUE = '\033[94m'
    __GREEN = '\033[92m'
    __RED = '\033[91m'
    __RED_BG = '\033[41m'
    __YELLOW = '\033[33m'
    __ENDC = '\033[0m'
    
    def _format_message_level(self, message, level):
        colors = {
            'INFO': self.__GREEN,
            'WARNING': self.__YELLOW,
            'ERROR': self.__RED,
            'CRITICAL': self.__RED_BG,
        }
        if level in colors.keys():
            message = "{color}{message}{ending}".format(color=colors[level],
                                                        message=message,
                                                        ending=self.__ENDC)
        return message
    
    def formatMessage(self, record):
        record.message = self._format_message_level(record.message, record.levelname)
        return super().formatMessage(record)
