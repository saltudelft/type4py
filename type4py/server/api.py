from flask import render_template, request, Blueprint, session
from type4py.server.app import app
from type4py.server.response import ServerResponse
from type4py.infer import PretrainedType4Py, type_annotate_file, get_type_checked_preds
from datetime import datetime

bp = Blueprint('type4py_api', __name__, template_folder='templates', url_prefix="/api/")

t4py_pretrained_m = None

@app.before_first_request
def load_type4py_model():
    global t4py_pretrained_m
    t4py_pretrained_m = PretrainedType4Py(app.config['MODEL_PATH'],
                                          app.config['DEVICE'],
                                          app.config['PRE_READ_TYPE_CLUSTER'])
    t4py_pretrained_m.load_pretrained_model()

@app.before_request
def before_request_f():
    session['req_start_t'] = datetime.now()

@bp.route('/')
def hello_world():
    return render_template('index.html')

@bp.route('/predict', methods = ['POST', 'GET'])
def upload():
    """
    POST method for uploading a file. Reads in a sent file and returns it.
    TODO: modify to your own needs
    """
    src_file = request.data

    if len(request.data.splitlines()) > app.config['MAX_LOC']:
        return ServerResponse(None, f"File is larger than {app.config['MAX_LOC']} LoC").get()
    
    if bool(int(request.args.get("tc"))):
        return ServerResponse(None, "Type-checking is not available yet!").get()
        #return ServerResponse(get_type_checked_preds(type_annotate_file(t4py_pretrained_m, src_file, None), src_file)).get()
    else:
        return ServerResponse(type_annotate_file(t4py_pretrained_m, src_file, None)).get()
