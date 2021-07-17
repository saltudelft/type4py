from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from os import getenv
import toml

app = Flask(__name__)

if getenv("FLASK_ENV") == "development":
    app.config.from_file("config_dev.toml", load=toml.load)
else:
    app.config.from_file("config.toml", load=toml.load)

app.secret_key = app.config['APP_SECRET_KEY']
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

if app.config['RATE_LIMIT']:
    limiter = Limiter(app,
                      default_limits=["5/hour", "100/day"], 
                      key_func=get_remote_address,
                      storage_uri='memcached://localhost:11211')

from type4py.server.api import bp
app.register_blueprint(bp)

from type4py.server.error_handlers import *
