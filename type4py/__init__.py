import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger_sh = logging.StreamHandler()
logger_sh.setLevel(logging.DEBUG)
logger_sh.setFormatter(logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(logger_sh)

__version__ = "0.1"
