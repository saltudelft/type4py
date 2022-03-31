from type4py.server.db import manager as dbm
from type4py.server import IS_T4PY_LOCAL_MODE
from flask import session
from flask_limiter.util import get_remote_address
from abc import ABC
from datetime import datetime
from hashlib import sha1


class PredictResponse:
    def __init__(self, response: dict, error: str=None) -> None:
        self.response = response
        self.error = error
    
    def get(self):
        self.log2db()
        if self.response is not None:
            self.response['session_id'] = session.get("session_id")
        return {'response': self.response, 'error': self.error}

    def log2db(self):
        if not IS_T4PY_LOCAL_MODE:
            dbm.sqla.session.add(dbm.PredictReqs(sha1(get_remote_address().encode()).hexdigest(), session.get("act_id"),
                                             session.get("session_id"), session.get('file_hash'), session.get("req_start_t"),
                                             datetime.now(), session.get('error'), self.response, session.get("ext_ver")))
            dbm.sqla.session.commit()


class TelemetryResponse(ABC):
    def __init__(self, *submit_data, response: str, error: str=None) -> None:
        pass

    def get(self):
        pass


class AcceptTypeResponse(TelemetryResponse):
    def __init__(self, *submit_data, response: str, error: str=None) -> None:
        self.submit_data = submit_data
        self.response = response
        self.error = error

    def get(self):
        if self.error is None:
            dbm.sqla.session.add(dbm.AcceptedTypes(*self.submit_data))
            dbm.sqla.session.commit()
        return {'response': self.response, 'error': self.error}


def is_session_id_valid(sess_token: str):
    return True if dbm.PredictReqs.query.filter_by(sess_id=sess_token).first() is not None else False
