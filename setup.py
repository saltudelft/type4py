from setuptools import setup
from os import path
from type4py import __version__

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='type4py',
    version=__version__,
    description='Type4Py: Deep Similarity Learning-Based Type Inference for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/saltudelft/type4py',
    author='Amir M. Mir (TU Delft)',
    author_email='mir-am@hotmail.com',
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Unix',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='deep learning type inference prediction similarity learning python source code type4py',
    packages=['type4py'],
    python_requries='>=3.5',
    install_requires=['numpy', 'tqdm', 'joblib', 'pandas', 'asttokens', 'astor', 'docstring-parser',
                      'scikit-learn', 'nltk', 'gensim', 'annoy', 'torch', 'dpu_utils', 'libsa4py'],
    entry_points={
        'console_scripts': [
            'type4py = type4py.__main__:main',
        ],
    }
)