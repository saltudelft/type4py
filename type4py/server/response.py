from type4py.server.db import manager as dbm
from flask import session
from flask_limiter.util import get_remote_address
from datetime import datetime
from hashlib import sha1

class ServerResponse:
    def __init__(self, response: dict, error: str=None) -> None:
        self.response = response
        self.error = error
    
    def get(self):
        dbm.sqla.session.add(dbm.PredictReqs(sha1(get_remote_address().encode()).hexdigest(), session.get("req_start_t"),
                                             datetime.now(), self.error))
        dbm.sqla.session.commit()
        return {'response': self.response, 'error': self.error}