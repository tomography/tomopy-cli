'''
    Log Lib for Sector 2-BM 
    
'''
import logging

# Logging defines
__GREEN = "\033[92m"
__RED = '\033[91m'
__YELLOW = '\033[33m'
__ENDC = '\033[0m'


logger = None
info_extra={'endColor': __ENDC, 'color': __GREEN}
warn_extra={'endColor': __ENDC, 'color': __YELLOW}
error_extra={'endColor': __ENDC, 'color': __RED}

def info(msg):
    global logger
    global info_extra
    logger.info(msg, extra=info_extra)

def error(msg):
    global logger
    global error_extra
    logger.error(msg, extra=error_extra)

def warning(msg):
    global logger
    global warn_extra
    logger.warning(msg, extra=warn_extra)


def setup_logger(log_name, stream_to_console=True):
    global logger
    global info_extra
    global warn_extra
    global error_extra

    info_extra['logger_name'] = log_name
    warn_extra['logger_name'] = log_name
    error_extra['logger_name'] = log_name
    logger = logging.getLogger(log_name)
    fHandler = logging.FileHandler(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(color)s  %(message)s %(endColor)s")
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    if stream_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)


