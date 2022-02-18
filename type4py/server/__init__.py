from type4py import logger
import os

logger.name = __name__
IS_T4PY_DOCKER_MODE = False

# If the Type4Py server is running inside Docker, intractions with the DB and
# the telemetry endpoint should be disabled.
if "T4PY_DOCKER_MODE" in os.environ:
    IS_T4PY_DOCKER_MODE = True
    logger.info("Running the Type4Py server in Docker mode")
