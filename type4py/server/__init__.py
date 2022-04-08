from type4py import logger
import os

logger.name = __name__
IS_T4PY_LOCAL_MODE = False

# If the Type4Py server is running inside Docker, intractions with the DB and
# the telemetry endpoint should be disabled.
if os.getenv("T4PY_LOCAL_MODE") == "1":
    IS_T4PY_LOCAL_MODE = True
