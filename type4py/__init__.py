import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger_sh = logging.StreamHandler()
logger_sh.setLevel(logging.DEBUG)
logger_sh.setFormatter(logging.Formatter(fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(logger_sh)

__version__ = "0.1"

# Constants and parameters
MIN_DATA_POINTS = 3
AVAILABLE_TYPES_NUMBER = 1024
KNN_TREE_SIZE = 20
MAX_PARAM_TYPE_DEPTH = 2
TOKEN_SEQ_LEN = (7, 3)
AVAILABLE_TYPE_APPLY_PROB = 0.5
