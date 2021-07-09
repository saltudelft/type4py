from flask import make_response
from type4py.server.app import app
from type4py.server.response import ServerResponse

@app.errorhandler(429)
def ratelimit_handler(e):
    return make_response(ServerResponse(None, "Ratelimit exceeded %s" % e.description).get(), 429)
