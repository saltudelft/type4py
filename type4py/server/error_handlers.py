from flask import make_response
from type4py.server.app import app
from type4py.server.response import ServerResponse
from type4py.server.app import app
from libcst._exceptions import ParserSyntaxError

@app.errorhandler(429)
def ratelimit_handler(e):
    return make_response(ServerResponse(None, "Ratelimit exceeded %s" % e.description).get(), 429)

@app.errorhandler(ParserSyntaxError)
def syntax_err_handler(e):
    return ServerResponse(None, "Could not parse the given source file! Check out its syntax.").get()
