'''
    Log Lib for Sector 2-BM 
    
'''
import logging

logger = logging.getLogger(__name__)

def setup_custom_logger(lfname, stream_to_console=True):
    fHandler = logging.FileHandler(lfname)
    logger.setLevel(logging.DEBUG)
    fformatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fHandler.setFormatter(fformatter)
    logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        cformatter = ColoredFormatter("%(asctime)s  %(levelname)s  %(message)s")
        ch.setFormatter(cformatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        self.__GREEN = "\033[92m"
        self.__RED = '\033[91m'
        self.__YELLOW = '\033[33m'
        self.__ENDC = '\033[0m'
        self.COLORS = {'WARNING': self.__YELLOW, 'INFO': self.__GREEN, 'DEBUG': self.__GREEN, 'CRITICAL': self.__RED, 'ERROR': self.__RED}
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        if self.use_color and record.levelname in self.COLORS:
            record.levelname = self.COLORS[record.levelname] + record.levelname + self.__ENDC 
        return logging.Formatter.format(self, record)