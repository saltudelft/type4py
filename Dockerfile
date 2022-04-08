FROM python:3.9
WORKDIR /type4py/
# Put the required models files in a folder "t4py_model_files" inside "/type4py"
# -type4py/
# --type4py/
# --t4py_model_files/
COPY . /type4py
ENV T4PY_LOCAL_MODE="1"

# The current model files are pickled with the below ver. of sklearn
RUN pip install scikit-learn==0.24.1

# Install Annoy from a pre-built binary wheel to avoid weird SIGILL error on some systems
RUN pip install https://type4py.com/pretrained_models/annoy-wheels/annoy-1.17.0-cp39-cp39-linux_x86_64.whl

RUN pip install -e .
# Web server's required packages
RUN pip install -r type4py/server/requirements.txt

# Install NLTK corpus
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('omw-1.4')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"

WORKDIR /type4py/type4py/server/

EXPOSE 5010

CMD ["bash", "run_server.sh"]