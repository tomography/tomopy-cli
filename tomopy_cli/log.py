'''
    Log Lib for Sector 2-BM 
    
'''
import logging

logger = logging.getLogger(__name__)

def info(msg):
    logger.info(msg)

def error(msg):
    logger.error(msg)

def warning(msg):
    logger.warning(msg)

def setup_custom_logger(lfname, stream_to_console=True):
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
    def __init__(self, fmt, datefmt=None, style='%'):
        # Logging defines
        self.__GREEN = "\033[92m"
        self.__RED = '\033[91m'
        self.__YELLOW = '\033[33m'
        self.__ENDC = '\033[0m'
        super().__init__(fmt, datefmt, style)
    
    
    def formatMessage(self,record):
        if record.levelname=='INFO':
            record.message = self.__GREEN + record.message + self.__ENDC
        elif record.levelname == 'WARNING':
            record.message = self.__YELLOW + record.message + self.__ENDC
        elif record.levelname == 'ERROR':
            record.message = self.__RED + record.message + self.__ENDC
        return super().formatMessage(record)
