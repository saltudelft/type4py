from flask import make_response
from type4py.server.app import app
from type4py.server.response import PredictResponse
from type4py.server.app import app
from libcst._exceptions import ParserSyntaxError

@app.errorhandler(429)
def ratelimit_handler(e):
    return make_response(PredictResponse(None, "Ratelimit exceeded %s" % e.description).get(), 429)

@app.errorhandler(ParserSyntaxError)
def syntax_err_handler(e):
    return PredictResponse(None, "Could not parse the given source file! Check out its syntax.").get()
