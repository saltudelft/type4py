from flask import Flask, render_template, request, Blueprint, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from type4py.infer import PretrainedType4Py, type_annotate_file, get_type_checked_preds

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

bp = Blueprint('type4py_api', __name__, template_folder='templates', url_prefix="/api/")

t4py_pretrained_m = None

class ServerResponse:
    def __init__(self, response: dict, error: str=None) -> None:
        self.response = response
        self.error = error
    
    def get(self):
        return {'response': self.response, 'error': self.error}


@bp.route('/')
def hello_world():
    return render_template('index.html')

@app.before_first_request
def load_type4py_model():
    global t4py_pretrained_m
    t4py_pretrained_m = PretrainedType4Py("/home/amir/MT4Py_typed_full/type4py_pretrained/")
    t4py_pretrained_m.load_pretrained_model()

@app.errorhandler(429)
def ratelimit_hander(e):
    return make_response(ServerResponse(None, "Ratelimit exceeded %s" % e.description).get(), 429)

@bp.route('/predict', methods = ['POST', 'GET'])
def upload():
    """
    POST method for uploading a file. Reads in a sent file and returns it.
    TODO: modify to your own needs
    """
    global t4py_pretrained_m
    src_file = request.data
    
    if bool(int(request.args.get("tc"))):
        print("Predictions with type-checking")
        return ServerResponse(get_type_checked_preds(type_annotate_file(t4py_pretrained_m, src_file, None), src_file)).get()
    else:
        print("Predictions without type-checking")
        return ServerResponse(type_annotate_file(t4py_pretrained_m, src_file, None)).get()

limiter = Limiter(app,
                  default_limits=["10/hour", "100/day"], 
                  key_func=get_remote_address,
                  storage_uri='memcached://localhost:11211')
app.register_blueprint(bp)
