#!/bin/bash

if [[ -z "${FLASK_ENV}" ]]; then
  # Production
  NoWorkers="${G_WORKERS:-2}"
  gunicorn -b 0.0.0.0:5010 -w "$NoWorkers" -k gevent wsgi:app --timeout 120
else
  # Development
  flask run -h 0.0.0.0 -p 5010
fi