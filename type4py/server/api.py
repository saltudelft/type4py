from flask import render_template, request, Blueprint, session
from type4py.server.app import app
from type4py.server.response import PredictResponse, AcceptTypeResponse, is_session_id_valid
from type4py.infer import PretrainedType4Py, type_annotate_file, get_type_checked_preds
from datetime import datetime
from secrets import token_urlsafe

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

    if request.endpoint == "type4py_api.predict":
        session['session_id'] = token_urlsafe(32)
        app.logger.info(f"Endpoint {request.endpoint}")

@bp.route('/')
def hello_world():
    return render_template('index.html')

@bp.route('/predict', methods = ['POST', 'GET'])
def predict():
    """
    POST method for uploading a file. Reads in a sent file and returns it.
    TODO: modify to your own needs
    """
    src_file = request.data
    is_fp_enabled = bool(int(request.args.get("fp"))) if request.args.get("fp") is not None else True
    session['file_hash'] = request.args.get('fh')
    session['ext_ver'] = request.args.get('ev')
    session['act_id'] = request.args.get('ai')

    # if len(request.data.splitlines()) > app.config['MAX_LOC']:
    #     return PredictResponse(None, f"File is larger than {app.config['MAX_LOC']} LoC").get()
    
    if bool(int(request.args.get("tc"))):
        return PredictResponse(None, "Type-checking is not available yet!").get()
        #return ServerResponse(get_type_checked_preds(type_annotate_file(t4py_pretrained_m, src_file, None), src_file)).get()
    else:
        return PredictResponse(type_annotate_file(t4py_pretrained_m, src_file, None, is_fp_enabled)).get()

@bp.route('/telemetry/accept_type', methods = ['GET'])
def submit_accepted_types():
    """
    Stores accepted types from the VSCode based on users' consent.
    """
    if is_session_id_valid(request.args.get('sid')):
        app.logger.info(f"Accepted type {request.args.get('at')} for {request.args.get('ts')} {request.args.get('idn')} at line {request.args.get('tsl')} with rank {request.args.get('r')} {int(request.args.get('cp'))} | sess: {request.args.get('sid')}")
        return AcceptTypeResponse(request.args.get('sid'), request.args.get('at'), request.args.get('r'), request.args.get('ts'), 
                                  request.args.get('idn'), request.args.get('tsl'), int(request.args.get('cp')),
                                  int(request.args.get('fp')), response='Thanks for submitting accepted types!').get()
    else:
        return AcceptTypeResponse(None, response=None, error="Invalid sessionID!").get()
