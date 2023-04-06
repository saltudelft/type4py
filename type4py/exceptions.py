class ModelNotFit(Exception):
    pass


class NotCompleteModel(ModelNotFit):
    def __init__(self):
        super().__init__("learn_split may just fit for complete model!")


class TrainedModel(Exception):
    pass


class ModelTrainedError(TrainedModel):
    def __init__(self):
        super().__init__("Model has been trained for this dataset!")


class EmdTypeError(Exception):
    pass


class EmdTypeNotFound(EmdTypeError):
    def __init__(self):
        super().__init__("Embedding Type not found!")


class TypeClusterNotFound(Exception):
    def __init__(self):
        super().__init__("Type clusters not found!")

class ModelNotfound(Exception):
    pass

class ModelNotExistsError(ModelNotfound):
    def __init__(self, model_name):
        super().__init__(f"Model {model_name} not found!")
