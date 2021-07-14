from type4py.server.db import manager as dbm
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
        dbm.sqla.session.add(dbm.PredictReqs(sha1(get_remote_address().encode()).hexdigest(), session.get("session_id"), session.get("req_start_t"),
                                             datetime.now(), self.error, self.response))
        dbm.sqla.session.commit()
        if self.response is not None:
            self.response['session_id'] = session.get("session_id")
        return {'response': self.response, 'error': self.error}


class TelemetryResponse(ABC):
    def __init__(self, *submit_data, response: str, error: str=None) -> None:
        pass

    def get(self):
        pass


class AcceptTypeResponse(TelemetryResponse):
    def __init__(self, *submit_data, response: str, error: str) -> None:
        self.submit_data = submit_data
        self.response = response
        self.error = error

    def get(self):
        dbm.sqla.session.add(dbm.AcceptedTypes(*self.submit_data))
        dbm.sqla.session.commit()
        return {'response': self.response, 'error': self.error}
