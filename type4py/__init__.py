import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger_sh = logging.StreamHandler()
logger_sh.setLevel(logging.DEBUG)
logger.addHandler(logger_sh)

logger_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(message)s')
logger_sh.setFormatter(logger_formatter)

__version__ = "0.1"
