from type4py.server.app import app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from enum import Enum

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{app.config['DB_USER']}:{app.config['DB_PASS']}@localhost:5432/{app.config['DB_NAME']}"
sqla = SQLAlchemy(app)
mig = Migrate(app, sqla)

class PredictReqs(sqla.Model):
    """
    Represents the table predict_reqs which stores requests to the server plus
    their timestamps and errors if any
    """
    __tablename__ = 'predict_reqs'

    id = sqla.Column(sqla.Integer, primary_key=True)
    hashed_IP = sqla.Column(sqla.String, nullable=False)
    start_t = sqla.Column(sqla.DateTime, nullable=False)
    finished_t = sqla.Column(sqla.DateTime, nullable=False)
    error = sqla.Column(sqla.String)

    def __init__(self, hashed_IP, start_t, finished_t, error):
        self.hashed_IP = hashed_IP
        self.start_t= start_t
        self.finished_t = finished_t
        self.error = error

class AcceptedTypes(sqla.Model):
    """
    Stores accepted types from users with their rank and type slot. 
    """

    class TypeSlots(Enum):
        Parameter = "Parameter"
        ReturnType = "ReturnType"
        Variable = "Variable"

    __tablename__ = 'accepted_types'

    id = sqla.Column(sqla.Integer, primary_key=True)
    accepted_type = sqla.Column(sqla.String, nullable=False)
    rank = sqla.Column(sqla.Integer, nullable=False)
    type_slot = sqla.Column(sqla.Enum(TypeSlots), nullable=False)
    filtered_preds = sqla.Column(sqla.Boolean, nullable=False)
    timestamp = sqla.Column(sqla.DateTime(timezone=True), server_default=sqla.func.now())

    def __init__(self, accepted_type, rank, type_slot, filtered_preds):
        self.accepted_type = accepted_type
        self.rank = rank
        self.type_slot = type_slot
        self.filtered_preds = filtered_preds
