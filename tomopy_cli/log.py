'''
    Log Lib for Sector 2-BM 
    
'''
import logging

# logger = setup_logger(__name__)
# logger = logging.getLogger(__name__)
# logger = logging.getLogger('test.txt')

# Logging defines
# __GREEN = "\033[92m"
# __RED = '\033[91m'
# __YELLOW = '\033[33m'
# __ENDC = '\033[0m'


# info_extra={'endColor': __ENDC, 'color': __GREEN}
# warn_extra={'endColor': __ENDC, 'color': __YELLOW}
# error_extra={'endColor': __ENDC, 'color': __RED}


# def info(msg):
#     logger.info(msg, extra=info_extra)

# def error(msg):
#     logger.error(msg, extra=error_extra)

# def warning(msg):
#     logger.warning(msg, extra=warn_extra)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    fhandler = logging.FileHandler(name)
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    chandler = logging.StreamHandler()
    chandler.setFormatter(formatter)
    chandler.setLevel(logging.DEBUG)
    logger.addHandler(chandler)

    return logger

def setup_old_logger(log_name, stream_to_console=True):

    logger = logging.getLogger(log_name)
    fHandler = logging.FileHandler(log_name)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(color)s  %(message)s %(endColor)s")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    return logger

