'''
Customized logging for the tomopy_cli library.

'''
import logging


logger = logging.getLogger(__name__)


def info(msg):
    logger.info(msg)


def error(msg):
    logger.error(msg)


def warning(msg):
    logger.warning(msg)


def setup_custom_logger(lfname: str=None, stream_to_console: bool=True):
    """Prepare the logging system with custom formatting.
    
    Parameters
    ----------
    lfname
      Path to where the log file should be stored. If omitted, no file
      logging will be performed.
    stream_to_console
      If true, logs will be output to the console with color
      formatting.
    
    """
    if lfname is not None:
        fHandler = logging.FileHandler(lfname)
        logger.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        fHandler.setFormatter(file_formatter)
        logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredLogFormatter('%(asctime)s - %(message)s'))
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)


class ColoredLogFormatter(logging.Formatter):
    """A logging formatter that add console color codes."""
    __GREEN = '\033[92m'
    __RED = '\033[91m'
    __YELLOW = '\033[33m'
    __ENDC = '\033[0m'
    
    def formatMessage(self, record):
        if record.levelname=='INFO':
            record.message = self.__GREEN + record.message + self.__ENDC
        elif record.levelname == 'WARNING':
            record.message = self.__YELLOW + record.message + self.__ENDC
        elif record.levelname == 'ERROR':
            record.message = self.__RED + record.message + self.__ENDC
        return super().formatMessage(record)
