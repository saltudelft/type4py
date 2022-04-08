from type4py.server import IS_T4PY_LOCAL_MODE
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from os import getenv, environ
import toml
import secrets

app = Flask(__name__)
app.config.from_file("config.toml", load=toml.load)

# Read default config values
if IS_T4PY_LOCAL_MODE:
    app.logger.info("Running the Type4Py server in local mode")
else:
    app.logger.info(f"Running the Type4Py server in {'production' if getenv('FLASK_ENV') is None else 'development'} mode...")
    # Read DB credentials
    if set(['T4Py_DB_ADDR', 'T4Py_DB_NAME', 'T4Py_DB_USER', 'T4Py_DB_PASS']).issubset(environ):
        app.config['DB_ADDR'] = getenv('T4Py_DB_ADDR')
        app.config['DB_NAME'] = getenv('T4Py_DB_NAME')
        app.config['DB_USER'] = getenv('T4Py_DB_USER')
        app.config['DB_PASS'] = getenv('T4Py_DB_PASS')
    else:
        raise RuntimeError("DB credentials not provided!")

# Overrides the default value of model's path and device provided in the config file
if getenv("T4Py_MODEL_PATH") is not None:
    app.config['MODEL_PATH'] = getenv("T4Py_MODEL_PATH")
if getenv("T4Py_DEVICE") is not None:
    app.config['DEVICE'] = getenv("T4Py_DEVICE")

app.secret_key = secrets.token_urlsafe(16)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if app.config['RATE_LIMIT']:
    limiter = Limiter(app,
                      default_limits=["5/hour", "100/day"], 
                      key_func=get_remote_address,
                      storage_uri='memcached://localhost:11211')

from type4py.server.api import bp
app.register_blueprint(bp)

from type4py.server.error_handlers import *
