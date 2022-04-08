from type4py.server.app import app
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from enum import Enum

app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{app.config['DB_USER']}:{app.config['DB_PASS']}@{app.config['DB_ADDR']}:5432/{app.config['DB_NAME']}"
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
    act_id = sqla.Column(sqla.String)
    sess_id = sqla.Column(sqla.String, nullable=False)
    file_hash = sqla.Column(sqla.String)
    start_t = sqla.Column(sqla.DateTime, nullable=False)
    finished_t = sqla.Column(sqla.DateTime, nullable=False)
    error = sqla.Column(sqla.String)
    extracted_features = sqla.Column(sqla.JSON)
    extension_ver = sqla.Column(sqla.String)

    def __init__(self, hashed_IP, act_id, sess_id, file_hash, start_t, finished_t,
                 error, extracted_features, extension_ver):
        self.hashed_IP = hashed_IP
        self.act_id = act_id
        self.sess_id = sess_id
        self.file_hash = file_hash
        self.start_t= start_t
        self.finished_t = finished_t
        self.error = error
        self.extracted_features = extracted_features
        self.extension_ver = extension_ver

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
    sess_id = sqla.Column(sqla.String, nullable=False)
    accepted_type = sqla.Column(sqla.String)
    rank = sqla.Column(sqla.Integer)
    type_slot = sqla.Column(sqla.Enum(TypeSlots, create_type=False), nullable=False)
    id_name = sqla.Column(sqla.String, nullable=False)
    id_ln = sqla.Column(sqla.Integer, nullable=False)
    canceled_preds = sqla.Column(sqla.Boolean, nullable=False)
    filtered_preds = sqla.Column(sqla.Boolean, nullable=False)
    timestamp = sqla.Column(sqla.DateTime(timezone=True), server_default=sqla.func.now())

    def __init__(self, sess_id, accepted_type, rank, type_slot,
                 id_name, id_ln, canceled_preds, filtered_preds):
        self.sess_id = sess_id
        self.accepted_type = accepted_type
        self.rank = rank
        self.type_slot = type_slot
        self.id_name = id_name
        self.id_ln = id_ln
        self.canceled_preds = canceled_preds
        self.filtered_preds = filtered_preds
