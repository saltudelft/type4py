import json

def read_file(filename: str) -> str:
    with open(filename) as file:
        return file.read()

def load_json(filename: str) -> dict:
    with open(filename, 'r') as json_f:
        return json.load(json_f)
