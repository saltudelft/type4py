from flask import make_response, session
from type4py.server.app import app
from type4py.server.response import PredictResponse
from type4py.server.app import app
from libsa4py.exceptions import ParseError
import traceback

@app.errorhandler(429)
def ratelimit_handler(e):
    session['error'] = str(e)
    return make_response(PredictResponse(None, "Ratelimit exceeded %s" % e.description).get(), 429)

@app.errorhandler(ParseError)
def syntax_err_handler(e):
    session['error'] = str(e)
    return PredictResponse(None, "Could not parse the given source file! Check out its syntax.").get()

@app.errorhandler(Exception)
def exception_handler(e):
    traceback.print_exc()
    session['error'] = str(e)
    return PredictResponse(None, "Could not predict types for the given source file!").get()
