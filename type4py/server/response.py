from type4py.server.db import manager as dbm
from flask import session
from flask_limiter.util import get_remote_address
from datetime import datetime

class ServerResponse:
    def __init__(self, response: dict, error: str=None) -> None:
        self.response = response
        self.error = error
    
    def get(self):
        dbm.sqla.session.add(dbm.PredictReqs(get_remote_address(), session.get("req_start_t"), datetime.now(), self.error))
        dbm.sqla.session.commit()
        return {'response': self.response, 'error': self.error}