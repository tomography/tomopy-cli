'''
    Log Lib for Sector 2-BM 
    
'''
import logging

# Logging defines
__GREEN = "\033[92m"
__RED = '\033[91m'
__YELLOW = '\033[33m'
__ENDC = '\033[0m'


info_extra={'endColor': __ENDC, 'color': __GREEN}
warn_extra={'endColor': __ENDC, 'color': __YELLOW}
error_extra={'endColor': __ENDC, 'color': __RED}

logger = logging.getLogger(__name__)

def info(msg):
    logger.info(info_extra['color']+ msg + info_extra['endColor'])

def error(msg):
    logger.error(error_extra['color']+ msg + error_extra['endColor'])

def warning(msg):
    logger.warning(warn_extra['color']+ msg + warn_extra['endColor'])


def setup_custom_logger(lfname, stream_to_console=True):

    fHandler = logging.FileHandler(lfname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)