from flask import render_template, request, Blueprint, session, jsonify, Response
from type4py.server import IS_T4PY_LOCAL_MODE
from type4py.server.app import app
from type4py.server.response import PredictResponse, AcceptTypeResponse, is_session_id_valid
from type4py.deploy.infer import PretrainedType4Py, type_annotate_file, get_type_checked_preds
from datetime import datetime
from secrets import token_urlsafe
import json

bp = Blueprint('type4py_api', __name__, template_folder='templates', url_prefix="/api/")

t4py_pretrained_m = None

with app.app_context():
    """
    load type4pymodel using app_context since Flask before_first_request has been
    deprecated https://github.com/pallets/flask/blob/main/CHANGES.rst#version-230
    """
    t4py_pretrained_m = PretrainedType4Py(app.config['MODEL_PATH'],
                                          app.config['DEVICE'],
                                          app.config['PRE_READ_TYPE_CLUSTER'],
                                          IS_T4PY_LOCAL_MODE)
    t4py_pretrained_m.load_pretrained_model()

@app.before_request
def before_request_f():
    session['req_start_t'] = datetime.now()
    if request.endpoint == "type4py_api.predict" or request.endpoint == "type4py_api.predict_fetch":
        session['session_id'] = token_urlsafe(32)
        app.logger.info(f"Endpoint {request.endpoint}")

@bp.route('/')
def hello_world():
    return render_template('index.html')

def predict_type4py_model(src_file: str, **args) -> dict:

    is_fp_enabled = bool(int(args.get("fp"))) if args.get("fp") is not None else True
    session['file_hash'] = args.get('fh')
    session['ext_ver'] = args.get('ev')
    session['act_id'] = args.get('ai')

    # if len(request.data.splitlines()) > app.config['MAX_LOC']:
    #     return PredictResponse(None, f"File is larger than {app.config['MAX_LOC']} LoC").get()

    if bool(int(args.get("tc"))):
        return PredictResponse(None, "Type-checking is not available yet!").get()
        #return ServerResponse(get_type_checked_preds(type_annotate_file(t4py_pretrained_m, src_file, None), src_file)).get()
    else:
        return PredictResponse(type_annotate_file(t4py_pretrained_m, src_file, None, is_fp_enabled)).get()


@bp.route('/predict', methods = ['POST', 'GET'])
def predict():
    """
    Queries the Type4Py model and returns the predicted type annotations as JSON.
    """
    return predict_type4py_model(request.data, tc=request.args.get("tc"), fp=request.args.get("fp"), fh=request.args.get('fh'),
                                 ev=request.args.get('ev'), ai=request.args.get('ai'))

@bp.route('/predict/fetch', methods = ['POST'])
def predict_fetch():
    """
    The predict endpoint to be used for the Fetch API
    """

    args = json.loads(request.data.decode('utf-8'))

    r = jsonify(predict_type4py_model(args['f'], **args))
    r.headers.add('Access-Control-Allow-Origin', '*')
    r.headers.add('Access-Control-Allow-Methods', 'POST, PUT, GET, OPTIONS')
    r.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization')
    return r


@bp.route('/telemetry/accept_type', methods = ['GET'])
def submit_accepted_types():
    """
    Stores accepted types from the VSCode based on users' consent.
    """

    if IS_T4PY_LOCAL_MODE:
        return Response(response="Telemetry is not supported when running the Type4Py server in the local mode", status=405)

    if is_session_id_valid(request.args.get('sid')):
        app.logger.info(f"Accepted type {request.args.get('at')} for {request.args.get('ts')} {request.args.get('idn')} at line {request.args.get('tsl')} with rank {request.args.get('r')} {int(request.args.get('cp'))} | sess: {request.args.get('sid')}")
        return AcceptTypeResponse(request.args.get('sid'), request.args.get('at'), request.args.get('r'), request.args.get('ts'),
                                  request.args.get('idn'), request.args.get('tsl'), int(request.args.get('cp')),
                                  int(request.args.get('fp')), response='Thanks for submitting accepted types!').get()
    else:
        return AcceptTypeResponse(None, response=None, error="Invalid sessionID!").get()
